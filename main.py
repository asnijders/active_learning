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
from src.utils import log_percentages, get_trainer, del_checkpoint, get_model, evaluate_model, train_model, create_project_filepath
from src.active import active

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", "Some weights of the model checkpoint")
os.environ["WANDB_SILENT"] = "true"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
wandb.login()


def main(args):

    # init logger
    wandb.init(config={'AL_iter': str(0),
                       'mode': 'FS' if config.seed_size == 1.0 else 'AL'})
    wandb.config.update(args)

    # initialise PL logging
    # project_filepath = create_project_filepath(config)
    logger = WandbLogger(project="active_learning",
                         save_dir=config.output_dir,
                         log_model=False)

    print('\n -------------- Active Learning -------------- \n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))

    # Set global seed
    seed_everything(config.seed, workers=True)
    if config.debug:  # This mode turns on more detailed torch error descriptions
        torch.autograd.set_detect_anomaly(True)

    if config.toy_run != 1.0:
        print('\nNOTE: TOY RUN - ONLY USING {}% OF DATA\n'.format(config.toy_run*100), flush=True)
    if config.downsample_rate != 1.0:
        print('\nNOTE: DOWN-SAMPLING - ONLY CONSIDERING {}% OF DATA\n'.format(config.downsample_rate * 100), flush=True)

    # --------------------------------------- Active learning: Seed phase ---------------------------------------------
    # Build datamodule
    dm = GenericDataModule(config=config)
    dm.setup(stage='fit')

    # log makeup of initial labelled pool
    log_percentages(mode='makeup',
                    new_indices=None,
                    logger=wandb,
                    dm=dm,
                    epoch=None)

    # --------------------------------------- Pool-based active learning ---------------------------------------------
    print('\nStarting Active Learning process with strategy: {}'.format(config.acquisition_fn))
    config.labelling_batch_size = dm.train.set_k(config.labelling_batch_size)
    for i in range(config.al_iterations):

        print('Active Learning iteration: {}\n'.format(i+1), flush=True)

        # ---------------------------------- Training model on current labelled dataset ------------------------------
        # initialise model
        model = get_model(config)

        # initialise trainer
        trainer = get_trainer(config=config,
                              logger=logger)

        # train model
        model, dm, logger, trainer = train_model(dm=dm,
                                                 config=config,
                                                 model=model,
                                                 logger=logger,
                                                 trainer=trainer)

        # ---------------------------------------------  Evaluating model  -------------------------------------------
        # evaluate best checkpoint on dev set
        evaluate_model(split='dev',
                       dm=dm,
                       config=config,
                       model=model,
                       trainer=trainer,
                       logger=wandb)

        # ------------------------------------ Acquiring new instances for labeling ----------------------------------
        # Exit AL loop if all data was already labelled
        if dm.has_unlabelled_data() is False:
            break

        # initialise acquisition function
        acquisition_fn = select_acquisition_fn(fn_id=config.acquisition_fn)

        # determine instances for labeling using provided acquisition fn
        to_be_labelled = acquisition_fn.acquire_instances(config=config,
                                                          model=model,
                                                          dm=dm,
                                                          k=config.labelling_batch_size)

        # log share of each dataset in set of queried examples
        log_percentages(mode='active',
                        new_indices=to_be_labelled,
                        logger=wandb,
                        dm=dm,
                        epoch=i)

        # label new instances
        dm.train.label_instances(to_be_labelled)

        # log composition of updated labelled pool
        log_percentages(mode='makeup',
                        new_indices=None,
                        logger=wandb,
                        dm=dm,
                        epoch=None)

        # delete now-redundant checkpoint
        del_checkpoint(trainer.checkpoint_callback.best_model_path)

        # some extra precautions to prevent memory leaks
        del model
        del trainer
        del acquisition_fn
        gc.collect()

    # --------------------------------------------- Testing final model --------------------------------------------- #
    # initialise final model
    model = get_model(config)

    # initialise trainer
    trainer = get_trainer(config=config,
                          logger=logger)

    # train model
    model, dm, logger, trainer = train_model(dm=dm,
                                             config=config,
                                             model=model,
                                             logger=logger,
                                             trainer=trainer)

    # evaluate on dev set(s)
    evaluate_model(split='dev',
                   dm=dm,
                   config=config,
                   model=model,
                   trainer=trainer,
                   logger=wandb)

    # evaluate on test set(s)
    evaluate_model(split='test',
                   dm=dm,
                   config=config,
                   model=model,
                   trainer=trainer,
                   logger=wandb)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Lisa args:
    parser.add_argument('--input_dir', default=None, type=str,
                        help='specifies where to read datasets from scratch disk')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='specifies where on scratch disk output should be stored')
    parser.add_argument('--checkpoint_dir', default=None, type=str,
                        help='specifies where on SCRATCH model checkpoints should be saved')
    parser.add_argument('--project_dir', default=None, type=str,
                        help='specify what project dir to write wandb experiment to')
    parser.add_argument('--array_uid', default=None, type=str,
                        help='unique ID of array job. useful for grouping multiple runs in wandb')

    # Experiment args
    parser.add_argument('--seed', default=42, type=int,
                        help='specifies global seed')

    parser.add_argument('--datasets', default='MNLI', type=str,
                        help='str to specify what nli datasets should be loaded'
                             'please separate datasets with a "," e.g.'
                             '--datasets="MNLI,SNLI,ANLI"')

    parser.add_argument('--undersample', action='store_false',
                        help='bool to specify whether majority training sets should be under sampled to'
                             'match minority datasets in number of examples'
                             'This results in a datapool where each sub-dataset is represented equally.')

    parser.add_argument('--seed_datasets', default=None, type=str,
                        help='str to specify which datasets should be used for initial seed'
                             'please separate datasets with a "," e.g.'
                             '--datasets="MNLI,SNLI,ANLI"')

    parser.add_argument('--checkpoint_datasets', default=None, type=str,
                        help='specify what dataset(s) should be used for dev-based checkpointing'
                        'multiple datasets: use aggregate of dev performances for checkpointing (not yet implemented)' 
                        'single dataset: only use a single dataset for dev-based checkpointing')

    parser.add_argument('--separate_test_sets', default=True, type=bool,
                        help='toggle between testing on:'
                             '- False: aggregate test set'
                             '- True: separate dataset-specific test sets'
                             'Note: any dataset used at train time will always be evaluated at dev/test time')

    parser.add_argument('--separate_eval_sets', default=True, type=bool,
                        help='toggle between evaluating (NOT checkpointing!) on:'
                             '- False: aggregate dev set'
                             '- True: separate dataset-specific dev sets')

    parser.add_argument('--model_id', default='bert-base-uncased', type=str,
                        help='specifies which transformer is used for encoding sentence pairs'
                             'Choose from: bert-base-uncased, bert-large-uncased, roberta-base')

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
    parser.add_argument('--al_iterations', default=10, type=int,
                        help='specifies number of active learning iterations')

    # Training args
    # TODO make sure that we use appropriate training parameters for transformers
    batch_size = 16 if device_count() > 0 else 4
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help='no. of sentences sampled per pass')
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='no. of epochs to train for')
    parser.add_argument('--patience', default=3, type=int,
                        help='patience for early stopping')
    parser.add_argument('--monitor', default='val_loss', type=str,
                        help='quantity monitored by early stopping / checkpointing callbacks.'
                             'choose between "val_loss" and "val_acc"')
    parser.add_argument('--lr', default=5e-6, type=float,
                        help='learning rate')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='hidden layer dropout prob')
    parser.add_argument('--max_length', default=350, type=int,
                        help='max no of tokens for tokenizer (default is enough for all tasks')

    # Lightning Trainer - distributed training args
    num_gpus = device_count() if device_count() > 0 else None
    parser.add_argument('--gpus', default=num_gpus, type=int)

    strategy = DDPStrategy(find_unused_parameters=False) if device_count() > 1 else None
    parser.add_argument('--strategy', default=strategy)

    accelerator = "gpu" if device_count() > 0 else None
    parser.add_argument('--accelerator', default=accelerator)

    num_workers = 3 if device_count() > 0 else 1  # TODO this may or may not lead to some speed bottlenecks
    parser.add_argument('--num_workers', default=num_workers, type=int,
                        help='no. of workers for DataLoaders')

    parser.add_argument('--precision', default=32, type=int,
                        help='sets floating point precision')

    # Auxiliary args
    parser.add_argument('--debug', default=False, type=bool,
                        help='toggle elaborate torch errors')
    parser.add_argument('--toy_run', default=1.00, type=float,
                        help='proportion of batches used in train/dev/test phase (useful for debugging)')
    parser.add_argument('--refresh_rate', default=250, type=int,
                        help='how often to refresh progress bar (in steps)')
    parser.add_argument('--progress_bar', action='store_false', help='toggle progress bar')

    log_every = 50 if device_count() > 0 else 1
    parser.add_argument('--log_every', default=log_every, type=int,
                        help='number of steps between logging')
    config = parser.parse_args()

    # convert CLI str-based dataset args to lists
    config.datasets = config.datasets.upper().replace(' ', '').split(',')

    # if no explicit seed dataset argument is given we assume uniform random sampling from entire pool of datasets
    if config.seed_datasets is None:
        config.seed_datasets = config.datasets
    else:
        config.seed_datasets = config.seed_datasets.upper().replace(' ', '').split(',')

    # convert seed and batch sizes to integers if need be
    if config.seed_size > 1:
        config.seed_size = int(config.seed_size)
    if config.labelling_batch_size > 1:
        config.labelling_batch_size = int(config.labelling_batch_size)

    if config.checkpoint_datasets is not None:
        config.checkpoint_datasets = config.checkpoint_datasets.upper().replace(' ', '').split()

        if len(config.checkpoint_datasets) > 1:
            raise NotImplementedError('Multi-dataset checkpointing is not implemented yet')

        if not set(config.datasets) >= set(config.checkpoint_datasets):
            raise ValueError('Please ensure that datasets used at checkpointing are at least subset of train datasets')

    if config.debug is True:
        config.al_iterations = 2

    main(config)
