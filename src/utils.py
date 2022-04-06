"""
This Python script will be used for any logic that does not belong to
a distinct component of the learning process
"""

from pytorch_lightning.utilities.memory import garbage_collection_cuda
import time
import gc
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.models import TransformerModel
import os


def create_project_filepath(config):
    """ Creates a logical filepath using provided experiment parameters """

    mode = ''
    if config.seed_size == 1.0:
        mode = 'full-supervision'
    else:
        mode = 'active-learning'

    filepath = 'project_{}/model_{}/data_{}/mode_{}'.format(config.project_dir,
                                                            config.model_id,
                                                            '-'.join(config.datasets),
                                                            mode)

    if config.checkpoint_datasets is not None:
        filepath += '/checkpoint-data_{}'.format('-'.join(config.checkpoint_datasets))

    return filepath


def log_results(logger, results, dm): # TODO put this in a Logger class

    # log training metrics
    results = results[0]
    for key in results.keys():
        res_dict = {'active_'+key: results[key], 'labelled_examples': len(dm.train.L)}
        logger.log(res_dict)

    return None


def get_model(config):
    """
    simple fn for initialising model
    """

    model = TransformerModel(model_id=config.model_id,
                             dropout=config.dropout,
                             lr=config.lr,
                             batch_size=config.batch_size,
                             acquisition_fn=config.acquisition_fn,
                             mc_iterations=config.mc_iterations,
                             num_gpus=config.gpus,
                             separate_test_sets=config.separate_test_sets)

    return model


def get_trainer(config, logger, batch_size=None, gpus=None):
    """
    simple fn for building trainer object
    :param gpus:
    :param config:
    :param logger:
    :param batch_size:
    :return:
    """

    if config.debug is False:

        mode = None
        if config.monitor == 'val_loss':
            mode = "min"
        elif config.monitor == 'val_acc':
            mode = "max"

        # Init early stopping
        early_stopping_callback = EarlyStopping(monitor=config.monitor,
                                                min_delta=0.00,
                                                patience=config.patience,
                                                verbose=True,
                                                mode=mode)

        # Init ModelCheckpoint callback, monitoring 'config.monitor'
        run_dir = config.checkpoint_dir + '/' + config.acquisition_fn + '/' + str(config.seed) + '/'
        checkpoint_callback = ModelCheckpoint(monitor=config.monitor,
                                              mode=mode,
                                              save_top_k=1,
                                              dirpath=run_dir,
                                              filename='{epoch}-{step}-{val_loss:.2f}-{val_acc:.2f}',
                                              verbose=True)

        callbacks = [early_stopping_callback, checkpoint_callback]
        epochs = config.max_epochs
        val_check_interval = config.val_check_interval

    else:
        callbacks = None
        epochs = 20
        val_check_interval = 1

    if gpus is None:
        gpus = config.gpus

    trainer = Trainer(gpus=gpus,
                      strategy=config.strategy,
                      logger=logger,
                      callbacks=callbacks,
                      log_every_n_steps=config.log_every,
                      accelerator=config.accelerator,
                      max_epochs=epochs,
                      min_epochs=2,
                      deterministic=True,
                      enable_checkpointing=True,
                      enable_model_summary=False,
                      val_check_interval=val_check_interval,
                      num_sanity_val_steps=1,
                      limit_val_batches=config.toy_run,
                      limit_train_batches=config.toy_run,
                      limit_test_batches=config.toy_run,
                      progress_bar_refresh_rate=config.refresh_rate,
                      enable_progress_bar=config.progress_bar,
                      precision=config.precision)

    return trainer


def train_model(dm, config, model, logger, trainer):
    """
    Trains provided model on labelled data, whilst checkpointing on one or more datasets
    :param dm: Data Module obj
    :param config: argparse obj
    :param model: TransformerModel instance
    :param logger: logger obj
    :param trainer: trainer obj
    :return: saved model with lowest dev-loss
    """

    # initialise dataloaders for current data
    labelled_loader = dm.labelled_dataloader()  # dataloader with labelled training data

    # one can either checkpoint based on single, or multiple-dataset dev performance
    if config.checkpoint_datasets is None:
        print('\nCheckpointing model weights based on aggregate dev performance on: {}'.format(dm.val.L['Dataset'].unique()))
        checkpoint_loader = dm.val_dataloader()

    else:
        print('\nCheckpointing model weights based on dev performance on: {}'.format(config.checkpoint_datasets))
        # dev loader for specified checkpointing dataset(s)
        checkpoint_loader = dm.get_separate_loaders(split='dev',
                                                    dataset_ids=config.checkpoint_datasets)

    # fine-tune model on (updated) labelled dataset L, from scratch, while checkpointing model weights
    print('\nFitting model on updated labelled pool, from scratch', flush=True)
    trainer.fit(model=model,
                train_dataloaders=labelled_loader,
                val_dataloaders=checkpoint_loader)

    # return model checkpoint with lowest dev loss
    if config.debug is False:
        print('\nLoading checkpoint: {}'.format(trainer.checkpoint_callback.best_model_path))
        model = TransformerModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model, dm, logger, trainer


def evaluate_model(dm, config, model, trainer, logger, split):
    """
    fn for evaluating trained model on either:
    - separate test sets, in case of multiple datasets
    - single test set, in case of aggregate or single test set
    :param split:
    :param logger:
    :param trainer:
    :param dm: datamodule obj
    :param config: argparse obj
    :param model: trained transformer instance
    :return: dictionary with test statistics
    """

    if split == 'test':
        if config.separate_test_sets is True:
            print('Evaluating best model on separate {} sets'.format(split), flush=True)
            test_loaders = dm.get_separate_loaders(split=split,
                                                   dataset_ids=dm.test.L['Dataset'].unique())

            for test_loader, dataset_id in zip(test_loaders, dm.test.L['Dataset'].unique()):


                model.test_set_id = dataset_id + '_'
                model.init_metrics()
                print('Test results for {}'.format(dataset_id), flush=True)
                results = trainer.test(model, test_loader)

                # log test results
                log_results(logger=logger,
                            results=results,
                            dm=dm)

            model.test_set_id = ''

        else:
            print('Evaluating best model on aggregate {} set'.format(split), flush=True)
            test_loader = dm.test_dataloader()
            model.init_metrics()
            results = trainer.test(model, test_loader)

            # log test results
            log_results(logger=logger,
                        results=results,
                        dm=dm)

    elif split == 'dev':
        if config.separate_eval_sets is True:
            print('Evaluating best model on separate {} sets'.format(split), flush=True)
            dev_loaders = dm.get_separate_loaders(split=split,
                                                  dataset_ids=dm.val.L['Dataset'].unique())

            for dev_loader, dataset_id in zip(dev_loaders, dm.val.L['Dataset'].unique()):

                # # TODO add a flag here to disable multi-dev-set evaluation for non-AL runs
                # if config.data_ratios is not None: # if this is true, we're doin a non-AL run.
                #
                #     if dataset_id != config.checkpoint_datasets[0]:
                #         print('Validation Warning: non-AL run - skipping evaluation on {} to save compute'.format(dataset_id), flush=True)
                #         continue

                model.dev_set_id = dataset_id + '_'
                model.init_metrics()
                print('Validation results for {}'.format(dataset_id), flush=True)
                results = trainer.validate(model, dev_loader)

                # log test results
                log_results(logger=logger,
                            results=results,
                            dm=dm)

            # reset dev set identifier to empty string
            model.dev_set_id = ''

        else:
            print('Evaluating best model on aggregate {} set'.format(split), flush=True)
            dev_loader = dm.val_dataloader()
            model.init_metrics()
            results = trainer.validate(model, dev_loader)

            # log test results
            log_results(logger=logger,
                        results=results,
                        dm=dm)


def log_percentages(mode, new_indices, logger, dm, epoch):
    """
    Logs the makeup of the current labelled pool, or the AL iteration, in terms of:
    - dataset composition
    - label distribution
    - more other things in the future?
    :param epoch:
    :param mode: 'makeup' logs statistics for the current pool of labelled examples
                 'active' logs statistics for AL iteration
    :param new_indices: set of indices of newly queried examples
    :param logger: wandb object
    :param dm: datamodule
    :return:
    """

    for key in ['Dataset', 'Label']:

        # monitor composition of labelled pool over time
        if mode == 'makeup':

            labelled_examples = dm.train.L
            percentages = labelled_examples[key].value_counts(normalize=True).to_dict()
            percentages = {k + '_makeup': v for k, v in percentages.items()}
            percentages['labelled_examples'] = len(dm.train.L)  # variable for x-axis: current L
            logger.log(percentages)

        # monitor composition of each AL batch
        elif mode == 'active':

            new_examples = dm.train.U.iloc[new_indices]  # select queried examples from unlabelled pool
            percentages = new_examples[key].value_counts(normalize=True).to_dict()
            percentages['labelled_examples'] = len(dm.train.L) + len(new_examples)  # variable for x-axis: old L + new batch
            percentages['AL_iter'] = epoch
            logger.log(percentages)

            # determine the composition of the current pool of unlabelled examples
            unlabelled_pool_composition = dm.train.U['Dataset'].value_counts(normalize=True).to_dict()

            # determine the composition of the most recent batch of queries
            new_batch_composition = new_examples['Dataset'].value_counts(normalize=True).to_dict()

            # normalise composition of new batch by correcting for size of sub-dataset in unlabelled pool
            normalised_proportions = {k + '_normalized': new_batch_composition[k]/unlabelled_pool_composition[k] for k,_ in new_batch_composition.items()}
            normalised_proportions['AL_iter'] = epoch
            normalised_proportions['labelled_examples'] = len(dm.train.L) + len(new_examples)  # TODO add a counter for the already labeled examples
            logger.log(normalised_proportions)

    return None


def del_checkpoint(filepath, verbose=True):

    try:
        os.remove(filepath)
        if verbose:
            print('Removed checkpoint at {}!'.format(filepath), flush=True)

    except Exception:
        pass


def collect_garbage():

    garbage_collection_cuda()
    time.sleep(5)
    torch.cuda.empty_cache()
    garbage_collection_cuda()
    gc.collect()
