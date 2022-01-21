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
import torch
from torch.cuda import device_count
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Local imports
from src.models import TransformerModel
from src.datasets import GenericDataModule
from src.strategies import select_acquisition_fn
from src.utils import log_results, c_print, log_percentages, get_trainer

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", "Some weights of the model checkpoint")
os.environ["WANDB_SILENT"] = "true"
wandb.login()


def main(args):

    c_print('\n -------------- Active Learning -------------- \n')
    # print CLI args
    c_print('Arguments: ')
    for arg in vars(args):
        c_print(str(arg) + ': ' + str(getattr(args, arg)))

    # Set global seed
    seed_everything(config.seed, workers=True)
    if config.debug:  # This mode turns on more detailed torch error descriptions
        torch.autograd.set_detect_anomaly(True)

    if config.toy_run != 1.0:
        c_print('\nNOTE: TOY RUN - ONLY USING {}% OF DATA\n'.format(config.toy_run*100), flush=True)
    if config.downsample_rate != 1.0:
        c_print('\nNOTE: DOWN-SAMPLING - ONLY CONSIDERING {}% OF DATA\n'.format(config.downsample_rate * 100), flush=True)

    # --------------------------------------- Active learning: Seed phase ---------------------------------------------
    # Build datamodule
    dm = GenericDataModule(config=config)
    dm.setup(stage='fit')

    # init logger
    wandb.init(config={'experiment_id' : str(config.uid),
                       'acquisition_fn': str(config.acquisition_fn),
                       'seed': str(config.seed),
                       'AL_iter': str(0)})

    # initialise PL logging and trainer objects
    logger = WandbLogger(project="active_learning",
                         save_dir=config.output_dir,
                         log_model=False)

    # log makeup of initial labelled pool
    log_percentages(mode='makeup',
                    new_indices=None,
                    logger=wandb,
                    dm=dm,
                    epoch=None)

    # --------------------------------------- Pool-based active learning ---------------------------------------------
    # TODO: abstract this code away into an active learner class, or...
    # TODO: wrap all of this in a function call such that everything gets cleaned up afterwards?

    c_print('\nStarting Active Learning process with strategy: {}'.format(config.acquisition_fn))
    config.labelling_batch_size = dm.train.set_k(config.labelling_batch_size)
    for i in range(config.iterations):

        c_print('Active Learning iteration: {}\n'.format(i), flush=True)

        # -----------------------------------  Fitting model on current labelled dataset ------------------------------
        # initialise model
        model = TransformerModel(model_id=config.model_id,
                                 dropout=config.dropout,
                                 lr=config.lr,
                                 batch_size=config.batch_size,
                                 acquisition_fn=config.acquisition_fn,
                                 mc_iterations=config.mc_iterations)
        # initialise trainer
        trainer = get_trainer(config=config,
                              logger=logger)

        # initialise dataloaders for current data
        labelled_loader = dm.labelled_dataloader()
        val_loader = dm.val_dataloader()

        # fine-tune model on updated labelled dataset L, from scratch
        c_print('\nFitting model on updated labelled pool, from scratch', flush=True)
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
            c_print('All examples were labelled. Terminating Active Learning Loop...')
            break

        if config.labelling_batch_size > len(dm.train.U):
            config.labelling_batch_size = len(dm.train.U)

        # determine instances for labeling using provided acquisition fn
        to_be_labelled = acquisition_fn.acquire_instances(config=config,
                                                          model=model,
                                                          dm=dm,
                                                          k=config.labelling_batch_size)

        # log share of each dataset in queried examples
        log_percentages(mode='active',
                        new_indices=to_be_labelled,
                        logger=wandb,
                        dm=dm,
                        epoch=i)

        # label new instances
        dm.train.label_instances(to_be_labelled)

        # log makeup of updated labelled pool
        log_percentages(mode='makeup',
                        new_indices=None,
                        logger=wandb,
                        dm=dm,
                        epoch=None)

        # some extra precautions to prevent memory leaks
        del model
        del trainer
        del acquisition_fn
        del labelled_loader
        del val_loader
        gc.collect()

    # --------------------------------------------- Testing final model --------------------------------------------- #
    # initialise final model
    model = TransformerModel(model_id=config.model_id,
                             dropout=config.dropout,
                             lr=config.lr,
                             batch_size=config.batch_size,
                             acquisition_fn=config.acquisition_fn,
                             mc_iterations=config.mc_iterations)

    trainer = get_trainer(config=config,
                          logger=logger)

    # initialise dataloaders for current data
    labelled_loader = dm.labelled_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # fine-tune model on final labelled dataset L, from scratch
    c_print('\nFitting model on updated labelled pool, from scratch', flush=True)
    trainer.fit(model=model,
                train_dataloaders=labelled_loader,
                val_dataloaders=val_loader)  # fit on labelled data

    # evaluate on test set and log statistics
    results = trainer.test(model, test_loader)
    log_results(logger=wandb,
                results=results,
                dm=dm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Lisa args:
    parser.add_argument('--input_dir', default=None, type=str,
                        help='specifies where to read datasets from scratch disk')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='specifies where on scratch disk output should be stored')
    parser.add_argument('--uid', default=None, type=str,
                        help='unique ID of array job. useful for grouping multiple runs in wandb')

    # Experiment args
    parser.add_argument('--seed', default=42, type=int,
                        help='specifies global seed')
    parser.add_argument('--datasets', default=['SNLI', 'ANLI', 'MNLI'], type=list,
                        help='list to specify nli datasets')
    parser.add_argument('--model_id', default='bert-base-uncased', type=str,
                        help='specifies which transformer is used for encoding sentence pairs'
                             'Choose from:'
                             'bert-base-uncased'
                             'bert-large-uncased')
    parser.add_argument('--acquisition_fn', default='coreset', type=str,
                        help='specifies which acquisition function is used for pool-based active learning')
    parser.add_argument('--mc_iterations', default=4, type=int,
                        help='specifies number of mc iterations over data')

    # Active Learning args
    parser.add_argument('--downsample_rate', default=1.00, type=float,
                        help='specifies fraction of data used for training')
    parser.add_argument('--seed_size', default=0.02, type=float,
                        help='specifies size of seed dataset')
    parser.add_argument('--labelling_batch_size', default=0.02, type=float,
                        help='specifies how many new instances will be labelled per AL iteration')
    parser.add_argument('--iterations', default=10, type=int,
                        help='specifies number of active learning iterations')
    # TODO add an arg to toggle downscaling of dev sets since this is done in some AL papers

    # Training args
    # TODO make sure that we use appropriate training parameters for transformers
    batch_size = 16 if device_count() > 0 else 4
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help='no. of sentences sampled per pass')
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='no. of epochs to train for')
    parser.add_argument('--patience', default=1, type=int,
                        help='patience for early stopping')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='hidden layer dropout prob')
    parser.add_argument('--max_length', default=350, type=int,
                        help='max no of tokens for tokenizer (default is enough for all tasks')

    # Lightning Trainer - distributed training args
    num_gpus = device_count() if device_count() > 0 else None
    parser.add_argument('--gpus', default=num_gpus)

    strategy = DDPStrategy(find_unused_parameters=False) if device_count() > 1 else None
    parser.add_argument('--strategy', default=strategy)

    accelerator = "gpu" if device_count() > 0 else None
    parser.add_argument('--accelerator', default=accelerator)

    num_workers = 3 if device_count() > 0 else 1  # TODO this may or may not lead to some speed bottlenecks
    parser.add_argument('--num_workers', default=num_workers, type=int,
                        help='no. of workers for DataLoaders')

    # Auxiliary args
    parser.add_argument('--debug', default=False, type=bool,
                        help='toggle elaborate torch errors')
    parser.add_argument('--toy_run', default=1.00, type=float,
                        help='proportion of batches used in train/dev/test phase (useful for debugging)')
    parser.add_argument('--refresh_rate', default=250, type=int,
                        help='how often to refresh progress bar (in steps)')

    log_every = 50 if device_count() > 0 else 1
    parser.add_argument('--log_every', default=log_every, type=int,
                        help='number of steps between loggings')
    config = parser.parse_args()

    main(config)
