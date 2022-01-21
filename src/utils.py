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
from pytorch_lightning.callbacks import EarlyStopping


def collect_garbage():
    garbage_collection_cuda()
    time.sleep(5)
    torch.cuda.empty_cache()
    garbage_collection_cuda()
    gc.collect()


# TODO put this in a Logger class
def log_results(logger, results, dm):

    # log training metrics
    results = results[0]
    for key in results.keys():
        res_dict = {'active_'+key: results[key], 'labelled_examples': len(dm.train.L)}
        logger.log(res_dict)

    return None


def get_trainer(config, logger, batch_size=None):
    """
    Factory for building trainer object
    :param config:
    :param logger:
    :param batch_size:
    :return:
    """

    if config.debug is False:
        early_stopping_callback = [EarlyStopping(monitor="val_loss",
                                                 min_delta=0.00,
                                                 patience=config.patience,
                                                 verbose=False,
                                                 mode="min")]
        epochs = config.max_epochs

    else:
        early_stopping_callback = None
        epochs = 1

    trainer = Trainer(gpus=config.gpus,
                      strategy=config.strategy,
                      logger=logger,
                      callbacks=early_stopping_callback,
                      log_every_n_steps=config.log_every,
                      accelerator=config.accelerator,
                      max_epochs=epochs,
                      deterministic=True,
                      enable_checkpointing=False,
                      enable_model_summary=False,
                      # profiler="simple",
                      num_sanity_val_steps=0,
                      limit_val_batches=config.toy_run,
                      limit_train_batches=config.toy_run,
                      limit_test_batches=config.toy_run,
                      progress_bar_refresh_rate=config.refresh_rate,
                      enable_progress_bar=True,
                      auto_scale_batch_size="binsearch")

    return trainer


def log_percentages(mode, new_indices, logger, dm, epoch):
    """
    Logs the makeup of the current labelled pool, or the AL iteration, in terms of:
    - dataset composition
    - label distribution
    - more other things in the future?
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


@rank_zero_only
def c_print(*args, **kwargs):

    return print(*args, **kwargs)
