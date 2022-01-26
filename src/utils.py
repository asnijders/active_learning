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


# TODO put this in a Logger class
def log_results(logger, results, dm):

    # log training metrics
    results = results[0]
    for key in results.keys():
        res_dict = {'active_'+key: results[key], 'labelled_examples': len(dm.train.L)}
        logger.log(res_dict)

    return None


def get_model(config):
    """simple fn for initialising model"""

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
        run_dir = config.checkpoint_dir + '/' + config.uid + '/' + config.acquisition_fn + '/' + str(config.seed) + '/'
        checkpoint_callback = ModelCheckpoint(monitor=config.monitor,
                                              mode=mode,
                                              save_top_k=1,
                                              dirpath=run_dir,
                                              filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
                                              verbose=True)

        callbacks = [early_stopping_callback, checkpoint_callback]
        epochs = config.max_epochs

    else:
        callbacks = None
        epochs = 1

    if gpus is None:
        gpus = config.gpus

    trainer = Trainer(gpus=gpus,
                      strategy=config.strategy,
                      logger=logger,
                      callbacks=callbacks,
                      log_every_n_steps=config.log_every,
                      accelerator=config.accelerator,
                      max_epochs=epochs,
                      deterministic=True,
                      enable_checkpointing=True,
                      enable_model_summary=False,
                      # profiler="simple",
                      num_sanity_val_steps=0,
                      limit_val_batches=config.toy_run,
                      limit_train_batches=config.toy_run,
                      limit_test_batches=config.toy_run,
                      progress_bar_refresh_rate=config.refresh_rate,
                      enable_progress_bar=True,
                      # auto_scale_batch_size="binsearch",
                      precision=config.precision)

    return trainer


def test_model(dm, config, model, trainer, logger):
    """
    fn for evaluating trained model on either:
    - separate test sets, in case of multiple datasets
    - single test set, in case of aggregate or single test set
    :param dm: datamodule obj
    :param config: argparse obj
    :param model: trained transformer instance
    :return: dictionary with test statistics
    """

    if config.separate_test_sets is True:
        print('Evaluating final model on separate test sets', flush=True)
        test_loaders = dm.get_separate_test_loaders()

        for test_loader, dataset_id in zip(test_loaders, config.datasets):
            model.test_set_id = dataset_id + '_'
            results = trainer.test(model, test_loader)

            # log test results
            log_results(logger=logger,
                        results=results,
                        dm=dm)

    else:
        print('Evaluating final model on aggregate test set', flush=True)
        test_loader = dm.test_dataloader()
        results = trainer.test(model, test_loader)

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

        if mode == 'makeup':

            labelled_examples = dm.train.L
            percentages = labelled_examples[key].value_counts(normalize=True).to_dict()
            percentages = {k + '_makeup': v for k, v in percentages.items()}
            percentages['labelled_examples'] = len(dm.train.L)  # variable for x-axis: current L
            logger.log(percentages)

        elif mode == 'active':

            new_examples = dm.train.U.iloc[new_indices]  # select queried examples from unlabelled pool
            percentages = new_examples[key].value_counts(normalize=True).to_dict()
            percentages['labelled_examples'] = len(dm.train.L) + len(new_examples)  # variable for x-axis: old L + new batch
            percentages['AL_iter'] = epoch
            logger.log(percentages)

    return None


def del_checkpoint(filepath):

    try:
        os.remove(filepath)
        print('Removed checkpoint at {}!'.format(filepath), flush=True)

    except Exception:
        pass


def cleanup():

    torch.distributed.destroy_process_group()


def collect_garbage():

    garbage_collection_cuda()
    time.sleep(5)
    torch.cuda.empty_cache()
    garbage_collection_cuda()
    gc.collect()
