"""
This module implements the following acquisition functions:

done:
- Random Sampling (baseline)
- Least Confidence
- Max Entropy
- Monte Carlo Max Entropy
- Bayesian Active Learning by Disagreement (BALD)

to do:
- Contrastive Acquisition
- Coresets

Credit:
- https://github.com/siddk/vqa-outliers
- https://github.com/rmunro/pytorch_active_learning/blob/master/uncertainty_sampling.py
- https://github.com/IBM/low-resource-text-classification-framework/blob/main/lrtc_lib/train_and_infer_service/train_and_infer_hf.py
"""

# global imports
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import entropy
import copy
from pytorch_lightning import Trainer
from src.utils import get_trainer
import time
import gc

# local imports
from src.utils import del_checkpoint
from src.coresets import CoresetGreedy, CoreSetMIPSampling
from src.discriminative_utils import DiscriminativeDataModule, DiscriminativeMLP, get_MLP_trainer


def get_predictions(datamodule, config, model):
    """
    simple fn for running inference on unlabelled data,
    for single-gpu or multi-gpu training
    multi-gpu: collect output from separate processes via all_gather and concatenate;
    single-gpu: just return predictions from trainer.predict()
    :param datamodule:
    :param config:
    :param model:
    :return:
    """

    if config.gpus > 1:

        dataloader = datamodule.unlabelled_dataloader()
        trainer = get_trainer(config, logger=None)
        _ = trainer.predict(model, dataloader)
        output = model.predictions.numpy()

    elif config.gpus == 1:

        dataloader = datamodule.unlabelled_dataloader()
        trainer = get_trainer(config, logger=None)
        predictions = trainer.predict(model, dataloader)
        predictions = torch.cat(predictions, dim=0)
        # print(predictions.size())
        output = predictions.numpy()


    return output


class AcquisitionFunction:
    def __init__(self, logger=None):

        pass

    def acquire_instances(self, config, model, dm, k):
        pass


class RandomSampler(AcquisitionFunction):
    def __init__(self, logger=None):
        super().__init__()

    def acquire_instances(self, config, model, dm, k):

        k_random_indices = np.random.choice(a=len(dm.train.U),
                                            size=int(k),
                                            replace=False)

        return k_random_indices


class LeastConfidence(AcquisitionFunction):

    def __init__(self, logger=None):
        super().__init__()

    def acquire_instances(self, config, model, dm, k):
        """
        This function implements least-confidence acquisition
        """

        predictions = get_predictions(datamodule=dm,
                                      config=config,
                                      model=model)

        max_probabilities = np.max(predictions, axis=1)
        probability_gap = 1 - np.array(max_probabilities)
        least_confident_indices = np.argsort(probability_gap)[::-1][:k]

        return least_confident_indices

        # with torch.no_grad():
        #     dataloader = dm.unlabelled_dataloader()
        #     print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
        #     model.cuda()
        #
        #     predictions = []
        #     for i, batch in enumerate(dataloader):
        #
        #         batch = {key: value.cuda() for key, value in batch.items()}
        #         prediction = model.active_step(batch, i)
        #         predictions.extend(prediction)
        #
        #     # we want to determine which examples have the largest probability gap
        #     # probability gap: 1 (absolute certainty) - the largest logit per prediction (=np.max(predictions, axis=1))
        #     predictions = np.asarray(predictions)
        #     max_probabilities = np.max(predictions, axis=1)
        #     probability_gap = 1 - np.array(max_probabilities)
        #     # we then determine the _indices_ that would sort this list of values in ascending order with np.argsort
        #     # we then reverse the list to descending order ([::-1])
        #     # we obtain the top k indices ([:k]), i.e. the examples with the highest probability gaps/lowest confidence
        #     least_confident_indices = np.argsort(probability_gap)[::-1][:k]

        # return least_confident_indices


class MaxEntropy(AcquisitionFunction):
    def __init__(self, mode, logger=None):
        super().__init__()
        self.mode = mode

    def acquire_instances(self, config, model, dm, k, dropout_k=10):
        """
        This function implements max-entropy and MC max-entropy acquisition
        """

        predictions = get_predictions(datamodule=dm,
                                      config=config,
                                      model=model)

        if self.mode == 'max-entropy':

            entropies = entropy(predictions, axis=1)
            max_entropy_indices = np.argsort(entropies)[::-1][:k]

            return max_entropy_indices

        elif self.mode == 'mc-max-entropy':

            mc_entropies = entropy(predictions, axis=1)
            mc_entropy_indices = np.argsort(mc_entropies)[::-1][:k]

            return mc_entropy_indices


class BALD(AcquisitionFunction):
    def __init__(self, logger=None):
        super().__init__()

    def acquire_instances(self, config, model, dm, k, dropout_k=10):
        """
        This function implements entropy uncertainty sampling acquisition
        """

        informations = get_predictions(datamodule=dm,
                                       config=config,
                                       model=model)

        bald_indices = np.argsort(informations)[::-1][:k]
        return bald_indices


class Coreset(AcquisitionFunction):
    # TODO NOTE: OLD IMPLEMENTATION
    # source: https://github.com/svdesai/coreset-al/blob/master/active_learn.py
    def __init__(self, logger=None):
        super().__init__()

    def acquire_instances(self, config, model, dm, k, dropout_k=10):

        def get_features(encoder, dataloader):
            """
            function for performing inference on labeled and unlabeled data
            takes a model and a dataloader, returns a list of embeddings
            """

            trainer = get_trainer(config, logger=None)
            _ = trainer.predict(encoder, dataloader)
            embeddings = encoder.predictions
            return list(embeddings)

            # features = []
            # encoder.eval()
            # encoder.cuda()
            #
            # with torch.no_grad():
            #     for i, batch in enumerate(loader):
            #
            #         # get representations in CLS space (following Active Learning for BERT: An Empirical Study)
            #         batch = {key: value.cuda() for key, value in batch.items()}
            #         embedding = encoder.embedding_step(batch=batch,
            #                                            batch_idx=i)
            #         # TODO add an alternative kind of sentence embedding?
            #         features.extend(embedding)
            #
            # encoder.train()
            #
            # return features

        # get features for unlabelled data
        unlabelled_features = get_features(encoder=model,
                                           dataloader=dm.unlabelled_dataloader())

        # get features for labelled data
        labelled_features = get_features(encoder=model,
                                         dataloader=dm.labelled_dataloader(shuffle=False))

        all_features = labelled_features + unlabelled_features

        labelled_indices = np.arange(0, len(labelled_features))

        coreset = CoresetGreedy(all_features)
        new_batch, max_distance = coreset.sample(already_selected=labelled_indices,
                                                 sample_size=k)

        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        new_batch = [i - len(labelled_features) for i in new_batch]

        return new_batch


class CoreSetLearner(AcquisitionFunction):

    def __init__(self, robustness_percentage=10 ** 4, max_to_consider=10 ** 6, greedy=False, logger=None):
        super().__init__()

        self.max_to_consider = max_to_consider
        self.greedy = greedy
        self.robustness_percentage = robustness_percentage

    def acquire_instances(self, config, model, dm, k, dropout_k=10):

        def get_features(encoder, dataloader):
            """
            function for performing inference on labeled and unlabeled data
            takes a model and a dataloader, returns a np array of embeddings
            """

            trainer = get_trainer(config, logger=None)
            predictions = trainer.predict(model, dataloader)
            predictions = torch.cat(predictions, dim=0)
            embeddings = predictions.squeeze(1).numpy()
            return embeddings

        # if self.max_to_consider and self.max_to_consider < len(unlabeled):
        #     unlabeled = unlabeled[:self.max_to_consider]

        # get range over length of labeled + unlabeled examples
        X_train = np.array(range(len(dm.train.L) + len(dm.train.U)))

        # get range over length of labeled examples
        labeled_idx = np.array(list(range(len(dm.train.L))))

        # get embeddings for labelled data
        labeled_embeddings = get_features(encoder=model,
                                          dataloader=dm.labelled_dataloader(shuffle=False))

        # get embeddings for unlabelled data
        unlabeled_embeddings = get_features(encoder=model,
                                            dataloader=dm.unlabelled_dataloader())

        # vertically concatenate embeddings
        embeddings = np.vstack((labeled_embeddings, unlabeled_embeddings))

        # initialise coreset sampler
        sampler = CoreSetMIPSampling(robustness_percentage=self.robustness_percentage,
                                     greedy=self.greedy)
        coreset_time = time.time()
        res = sampler.query(X_train=X_train,
                            labeled_idx=labeled_idx,
                            amount=k,
                            representation=embeddings)

        print('Elapsed time for coreset selection: {} seconds'.format(time.time() - coreset_time), flush=True)
        # print(res, flush=True)

        # filter out all indices that correspond to labeled indice
        # selected_idx = np.sort([idx for idx in res if idx not in set(labeled_idx)])
        return res.tolist()


class DiscriminativeActiveLearner(AcquisitionFunction):
    def __init__(self, logger):
        super().__init__()
        self.sub_queries = 10
        self.logger = logger

    def train_discriminator(self, config, logger, discriminative_dm):

        # init MLP discriminator model
        model = DiscriminativeMLP(batch_size=64)

        # init trainer obj
        trainer = get_MLP_trainer(config=config, logger=logger)

        # init dataloader for unlabeled + labeled
        train_loader = discriminative_dm.train_loader()

        # train MLP discriminator on unlabeled + labeled data
        trainer.fit(model=model, train_dataloaders=train_loader)

        # load model with best train accuracy
        print('\nLoading checkpoint for discriminator: {}'.format(trainer.checkpoint_callback.best_model_path),
              flush=True)
        model = DiscriminativeMLP.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        return model, trainer

    def get_predictions(self, model, discriminative_dm, trainer):

        # init dataloader for unlabeled data
        unlabeled_loader = discriminative_dm.unlabeled_loader()

        # run inference on unlabeled data
        predictions = trainer.predict(model, unlabeled_loader)

        # concatenate list of arrays to single array along batch dim
        predictions = np.concatenate(predictions, axis=0)

        # we are interested in examples where the model is most confident about them being UNLABELED (index = 0)
        predictions = predictions[:, 0]  # select all rows, first column
        predictions = np.argsort(predictions)[::-1]  # sort model confidence in descending order

        return predictions

    def acquire_instances(self, config, model, dm, k):

        # timeit
        dal_start = time.time()

        # construct dataset for learned representations
        discriminative_dm = DiscriminativeDataModule(config=config,
                                                     model=model,
                                                     dm=dm,
                                                     disc_batch_size=64)

        # define acquisition parameters
        sub_sample_size = int(k/self.sub_queries)
        labeled_so_far = 0
        iteration = 0
        already_seen = []

        # loop until k examples have been labeled
        while labeled_so_far < k:
            if labeled_so_far + sub_sample_size > k:
                sub_sample_size = k - labeled_so_far

            iteration += 1

            # train discriminator on unlabeled + labeled data
            model, trainer = self.train_discriminator(config=config,
                                                      logger=self.logger,
                                                      discriminative_dm=discriminative_dm)

            # perform inference on unlabeled data;
            # obtain predictions for which the model is most confident of them being unlabeled
            predictions = self.get_predictions(model=model,
                                               trainer=trainer,
                                               discriminative_dm=discriminative_dm)

            # Problem: per sub-batch, I need to move the newly selected indices from the unlabeled pool since we know
            # that they will be labeled in the future (so we can already assume them to be labeled for the next
            # DAL iteration. if I move them and reset the index for the unlabeled ex., the indices no longer correspond
            # to the underlying unlabelled data pool when we're _actually_ going to label the new batch!
            # one way to circumvent is to keep track of which examples we plan on labelling. in the meantime, in the
            # 'unlabeled' pool I can toggle the labels to 1, so that when the model gets trained it now considers them
            # as 'labeled' examples. In the meantime, I don't change anything in the unlabeled pool.
            # When I then select indices for the next DAL iteration, I can check whether I already picked them before,
            # and if so I can just skip them.

            # first return a list of the indices corresponding to the sorted predictions from hi to low
            # then fill up the sub_query
            sub_batch = []  # keep track of indices for sub_query
            for index in predictions:
                if index not in already_seen:  # check whether an index was already seen during a previous round
                    sub_batch.append(index)
                if len(sub_batch) >= sub_sample_size:  # stop when enough examples for sub-query have been accumulated
                    break

            # label new sub-batch of unlabeled examples in _training_ set
            discriminative_dm.train.label_instances(sub_batch)

            # add these indices to running list in case we encounter them again in next round
            already_seen.extend(sub_batch)

            # add ex. counter
            labeled_so_far += len(sub_batch)

            # delete checkpoint for this sub-round
            del_checkpoint(trainer.checkpoint_callback.best_model_path, verbose=False)
            del model
            del trainer
            gc.collect()

            print('DAL Iteration: {}. Labeled {} new examples.'.format(iteration,
                                                                       len(sub_batch), flush=True))

        # 3. return complete acquired batch
        print('Time needed for {} sub-queries for {} examples: {} seconds'.format(iteration,
                                                                                  k,
                                                                                  time.time()-dal_start),
              flush=True)

        assert len(already_seen) == len(set(already_seen))
        return already_seen


def select_acquisition_fn(fn_id, logger):
    """
    This function takes a acquisition function id and returns
    an instance of the corresponding AcquisitionFunction object
    :param fn_id:
    :return:
    """
    acquisition_fn = None

    if fn_id == 'random':
        acquisition_fn = RandomSampler()

    elif fn_id == 'least-confidence':
        acquisition_fn = LeastConfidence()

    elif fn_id == 'max-entropy':
        acquisition_fn = MaxEntropy(mode='max-entropy')

    elif fn_id == 'mc-max-entropy':
        acquisition_fn = MaxEntropy(mode='mc-max-entropy')

    elif fn_id == 'bald':
        acquisition_fn = BALD()

    elif fn_id == 'coreset':
        acquisition_fn = CoreSetLearner(greedy=True)

    elif fn_id == 'badge':
        raise NotImplementedError

    elif fn_id == 'alps':
        raise NotImplementedError

    elif fn_id == 'contrastive':
        raise NotImplementedError

    elif fn_id == 'cartography':
        raise NotImplementedError

    elif fn_id == 'dal':
        acquisition_fn = DiscriminativeActiveLearner(logger=logger)

    else:
        raise KeyError('No acquisition function found for {}'.format(fn_id))

    return acquisition_fn
