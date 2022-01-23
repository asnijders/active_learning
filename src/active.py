# General imports
import argparse
import sys
import os
import wandb
import gc
import warnings
import logging
import datetime
import time

# Torch modules

# Local imports
from src.models import TransformerModel
from src.datasets import GenericDataModule
from src.strategies import select_acquisition_fn
from src.utils import log_results, log_percentages, get_trainer


def active(i, config, dm, logger):

    print('Active Learning iteration: {}\n'.format(i), flush=True)

    # -----------------------------------  Fitting model on current labelled dataset ------------------------------
    # initialise model
    model = TransformerModel(model_id=config.model_id,
                             dropout=config.dropout,
                             lr=config.lr,
                             batch_size=config.batch_size,
                             acquisition_fn=config.acquisition_fn,
                             mc_iterations=config.mc_iterations,
                             num_gpus=config.gpus)

    print('initialised model', flush=True)

    # initialise trainer
    trainer = get_trainer(config=config,
                          logger=logger)

    # initialise dataloaders for current data
    labelled_loader = dm.labelled_dataloader()
    val_loader = dm.val_dataloader()

    # fine-tune model on updated labelled dataset L, from scratch
    print('\nFitting model on updated labelled pool, from scratch', flush=True)
    trainer.fit(model=model,
                train_dataloaders=labelled_loader,
                val_dataloaders=val_loader)  # fit on labelled data

    # evaluate and log results for current examples
    results = trainer.validate(model, val_loader)
    log_results(logger=wandb,
                results=results,
                dm=dm)

    # ----------------------------------- Acquiring new instances for labeling -----------------------------------
    # set training dataset mode to access the unlabelled data
    dm.train.set_mode('U')

    # initialise acquisition function
    acquisition_fn = select_acquisition_fn(fn_id=config.acquisition_fn)

    if len(dm.train.U) == 0:
        print('All examples were labelled. Terminating Active Learning Loop...')
        return None

    if config.labelling_batch_size > len(dm.train.U):
        config.labelling_batch_size = len(dm.train.U)

    # determine instances for labeling using provided acquisition fn
    to_be_labelled = acquisition_fn.acquire_instances(config=config,
                                                      model=model,
                                                      dm=dm,
                                                      k=config.labelling_batch_size)

    return to_be_labelled

    # # log share of each dataset in queried examples
    # log_percentages(mode='active',
    #                 new_indices=to_be_labelled,
    #                 logger=wandb,
    #                 dm=dm,
    #                 epoch=i)
    #
    # label new instances

    #
    # # log makeup of updated labelled pool
    # log_percentages(mode='makeup',
    #                 new_indices=None,
    #                 logger=wandb,
    #                 dm=dm,
    #                 epoch=None)

    # del model
    # del trainer
    # del acquisition_fn
    # del labelled_loader
    # del val_loader
    # gc.collect()