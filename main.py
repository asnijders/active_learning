# General imports
import argparse
import sys
import time
import os
import wandb

# PyTorch modules
import torch
from torch.cuda import device_count
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

# Local imports
from src.models import TransformerModel
from src.datasets import GenericDataModule
from src.strategies import select_acquisition_fn

import warnings
import logging
# logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", ".*does not have many workers.*")
wandb.login()


def main(args):

    print('\n -------------- Active Learning -------------- \n')
    # print CLI args
    print('Arguments: ')
    for arg in vars(args):
        print(str(arg) + ': ' + str(getattr(args, arg)))
    # Set global seed
    seed_everything(config.seed)
    if config.debug: # This mode turns on more detailed torch error descriptions
        torch.autograd.set_detect_anomaly(True)

    # --------------------------------------- Active learning: Seed phase ---------------------------------------------
    # Build datamodule
    dm = GenericDataModule(datasets=config.datasets,
                           seed_size=config.seed_size,
                           max_length=config.max_length,
                           batch_size=config.batch_size,
                           num_workers=config.num_workers,
                           input_dir=config.input_dir)
    dm.prepare_data()
    # dm.setup(stage="fit")

    # initialise model
    model = TransformerModel(model_id=config.model_id,
                             dropout=config.dropout,
                             lr=config.lr,
                             batch_size=config.batch_size)

    # initialise PL logging and trainer objects
    seed_logger = WandbLogger(project="active_learning_seed")
    seed_trainer = Trainer(gpus=config.gpus,
                           logger=seed_logger,
                           log_every_n_steps=config.log_every,
                           accelerator=config.accelerator,
                           max_epochs=config.max_epochs,
                           enable_model_summary=False,
                           limit_val_batches=1,
                           num_sanity_val_steps=1)

    # Fine-tune model on initial seed set
    seed_trainer.fit(model, dm)
    wandb.finish()
    print('Finished fine-tuning model on initial seed!')

    # --------------------------------------- Pool-based active learning ---------------------------------------------

    # TODO: abstract this code away into an active learner class
    print('\nStarting Active Learning process with strategy: {}'.format(config.acquisition_fn))

    # initialise acquisition function
    acquisition_fn = select_acquisition_fn(config.acquisition_fn)

    # define new trainer and logger specific to active learning phase
    active_logger = WandbLogger(project="active_learning_iter")
    active_trainer = Trainer(gpus=config.gpus,
                             logger=active_logger,
                             log_every_n_steps=config.log_every,
                             accelerator=config.accelerator,
                             max_epochs=config.max_epochs,
                             limit_val_batches=1,
                             enable_model_summary=False)

    # start acquisition loop
    for i in range(config.iterations):

        print('Active Learning iteration: {}'.format(i+1))

        # set training dataset mode to access the unlabelled data
        dm.train.set_mode('U')

        # acquire new examples for labelling
        to_be_labelled = acquisition_fn.acquire_instances(config=config,
                                                          model=model,
                                                          dm=dm,
                                                          k=config.labelling_batch_size)

        # label new instances and move from U to L
        dm.train.label_instances(to_be_labelled)

        # re-set training dataset mode to access the labelled data
        dm.train.set_mode('L')

        # initialize a new model
        model = TransformerModel(model_id=config.model_id,
                                 dropout=config.dropout,
                                 lr=config.lr,
                                 batch_size=config.batch_size)

        # fine-tune model on updated labelled dataset L, from scratch
        active_trainer.fit(model, dm)

    # run test set
    dm.setup(stage="test")
    result = active_trainer.test(model=model,
                          datamodule=dm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Lisa args:

    parser.add_argument('--input_dir', default=None, type=str,
                        help='specifies where to read datasets from scratch disk')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='specifies where on scratch disk output should be stored')

    # Program args
    parser.add_argument('--seed', default=42, type=int,
                        help='specifies global seed')
    parser.add_argument('--datasets', default=['ANLI', 'SNLI'], type=list,
                        help='list to specify nli datasets')
    parser.add_argument('--model_id', default='bert-base-uncased', type=str,
                        help='specifies which transformer is used for encoding sentence pairs')

    # Active Learning args
    parser.add_argument('--seed_size', default=0.10, type=float, # TODO change this to percentage of dataset size later
                        help='specifies size of seed dataset')
    parser.add_argument('--acquisition_fn', default='coreset', type=str,
                        help='specifies which acquisition function is used for pool-based active learning')
    parser.add_argument('--iterations', default=5, type=int,
                        help='specifies number of active learning iterations')
    parser.add_argument('--labelling_batch_size', default=10, type=int, #TODO change this to percentage of dataset size
                        help='specifies how many new instances will be labelled per AL iteration')

    # Training args
    # TODO make sure that we use appropriate training parameters for transformers
    batch_size = 32 if device_count() > 0 else 4
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help='no. of sentences sampled per pass')
    parser.add_argument('--max_epochs', default=1, type=int,
                        help='no. of epochs to train for')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='learning rate')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='hidden layer dropout prob')
    parser.add_argument('--max_length', default=180, type=int, #TODO make sure this param is set appropriately
                        help='max no of tokens for tokenizer (default is enough for all tasks')

    # Lightning Trainer args
    num_gpus = device_count() if device_count() > 0 else None # TODO update this value once everything work such that we can utilize all GPUs
    parser.add_argument('--gpus', default=num_gpus)

    accelerator = None if device_count() > 0 else None
    parser.add_argument('--accelerator', default=accelerator)

    num_workers = os.cpu_count() if device_count() > 0 else 1 # TODO this may or may not lead to some speed bottlenecks
    parser.add_argument('--num_workers', default=num_workers, type=int,
                        help='no. of workers for DataLoaders')

    log_every = 10 if device_count() > 0 else 1
    parser.add_argument('--log_every', default=log_every, type=int,
                        help='number of steps between loggings')

    # Auxiliary args
    parser.add_argument('--debug', default=True, type=bool,
                        help='toggle elaborate torch errors')
    parser.add_argument('--toy_run', default=1, type=int,
                        help='set no of batches per datasplit per epoch (helpful for debugging)')
    config = parser.parse_args()

    main(config)
