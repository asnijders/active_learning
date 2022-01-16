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
from src.utils import c_print

# local imports
from src.coresets import CoresetGreedy


class AcquisitionFunction:
    def __init__(self):

        pass

    def acquire_instances(self, config, model, dm, k):
        pass


class RandomSampler(AcquisitionFunction):
    def __init__(self):
        super().__init__()

    def acquire_instances(self, config, model, dm, k):

        k_random_indices = np.random.choice(a=len(dm.train.U),
                                            size=int(k),
                                            replace=False)

        return k_random_indices


class LeastConfidence(AcquisitionFunction):

    def __init__(self):
        super().__init__()

    def acquire_instances(self, config, model, dm, k):
        """
        This function implements least-confidence acquisition
        """

        with torch.no_grad():
            dataloader = dm.unlabelled_dataloader()
            c_print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
            model.cuda()

            predictions = []
            for i, batch in enumerate(dataloader):

                batch = {key: value.cuda() for key, value in batch.items()}
                prediction = model.active_step(batch, i)
                predictions.extend(prediction)

            # we want to determine which examples have the largest probability gap
            # probability gap: 1 (absolute certainty) - the largest logit per prediction (=np.max(predictions, axis=1))
            predictions = np.asarray(predictions)
            max_probabilities = np.max(predictions, axis=1)
            probability_gap = 1 - np.array(max_probabilities)
            # we then determine the _indices_ that would sort this list of values in ascending order with np.argsort
            # we then reverse the list to descending order ([::-1])
            # we obtain the top k indices ([:k]), i.e. the examples with the highest probability gaps/lowest confidence
            least_confident_indices = np.argsort(probability_gap)[::-1][:k]

        return least_confident_indices


class MaxEntropy(AcquisitionFunction):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def acquire_instances(self, config, model, dm, k, dropout_k=10):
        """
        This function implements max-entropy and MC max-entropy acquisition
        """

        # keep track of most probable label probabilities
        max_entropies = []

        with torch.no_grad():

            dataloader = dm.unlabelled_dataloader()
            c_print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
            model.cuda()

            if self.mode == 'max-entropy':

                entropies = []
                for i, batch in enumerate(dataloader):

                    batch = {key: value.cuda() for key, value in batch.items()}
                    prediction = model.active_step(batch, i)
                    entropies.extend(list(entropy(prediction, axis=1)))

                # given the list of entropies, we
                # 1) determine the indices that sort the array in ascending order
                # 2) revert the list to descending order (high to low)
                # 3) take the top k indices from the reverted list to obtain examples with max entropy
                entropies = np.asarray(entropies)
                max_entropy_indices = np.argsort(entropies)[::-1][:k]

                return max_entropy_indices

            elif self.mode == 'mc-max-entropy':

                mc_entropies = []
                for i, batch in enumerate(dataloader):
                    batch = {key: value.cuda() for key, value in batch.items()}
                    mc_prediction = model.mc_step(batch, i)
                    mc_entropy = entropy(mc_prediction, axis=1)
                    mc_entropies.extend(list(mc_entropy))

                mc_entropy_indices = np.argsort(mc_entropies)[::-1][:k]

                return mc_entropy_indices


class BALD(AcquisitionFunction):
    def __init__(self):
        super().__init__()

    def acquire_instances(self, config, model, dm, k, dropout_k=10):
        """
        This function implements entropy uncertainty sampling acquisition
        """

        with torch.no_grad():

            dataloader = dm.unlabelled_dataloader()
            c_print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
            model.cuda()

            informations = []
            for i, batch in enumerate(dataloader):

                batch = {key: value.cuda() for key, value in batch.items()}
                information = model.bald_step(batch, i)
                informations.extend(information)

            bald_indices = np.argsort(informations)[::-1][:k] #TODO check correctness

        return bald_indices


class Coreset(AcquisitionFunction):
    # source: https://github.com/svdesai/coreset-al/blob/master/active_learn.py
    def __init__(self):
        super().__init__()

    def acquire_instances(self, config, model, dm, k, dropout_k=10):

        def get_features(encoder, loader):
            """
            function for performing inference on labeled and unlabeled data
            takes a model and a dataloader, returns a list of embeddings
            """
            features = []
            encoder.eval()
            encoder.cuda()

            with torch.no_grad():
                for i, batch in enumerate(loader):

                    # get representations in CLS space (following Active Learning for BERT: An Empirical Study)
                    batch = {key: value.cuda() for key, value in batch.items()}
                    embedding = encoder.embedding_step(batch=batch,
                                                       batch_idx=i)
                    # TODO add an alternative kind of sentence embedding?
                    features.extend(embedding)

            encoder.train()

            return features

        # create separate dataset objects for labeled and unlabeled data
        # create dataloader for unlabelled data
        unlabelled_loader = dm.unlabelled_dataloader()

        # get features for unlabelled data
        unlabelled_features = get_features(encoder=model,
                                           loader=unlabelled_loader)

        # create dataloader for labelled data
        labelled_loader = dm.labelled_dataloader(shuffle=False)

        # get features for labelled data
        labelled_features = get_features(encoder=model,
                                         loader=labelled_loader)

        all_features = labelled_features + unlabelled_features

        labelled_indices = np.arange(0, len(labelled_features))

        coreset = CoresetGreedy(all_features)
        new_batch, max_distance = coreset.sample(already_selected=labelled_indices,
                                                 sample_size=k)

        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        new_batch = [i - len(labelled_features) for i in new_batch]

        return new_batch


# def get_predictions(model, dm, config, trainer):
#     """
#     This function performs inference using a trained model in Pytorch Lightning
#     :param trainer:
#     :param model:
#     :param dm:
#     :param config:
#     :return: list of predictions
#     """
#
#     # TODO make sure that dropout is still enabled for MC-based strategies (unsure whether this is disabled for train.predict)
#     # TODO make sure that all strategies using dataloaders only use NON SHUFFLED dataloaders to preserve meaningful order of examples and indices
#
#     # TODO see below
#     """
#     The BasePredictionWriter should be used while using a spawn based accelerator.
#     This happens for Trainer(strategy="ddp_spawn") or training on
#     8 TPU cores with Trainer(tpu_cores=8) as predictions won’t be returned.
#     """
#
#
#     dataloader = dm.unlabelled_dataloader()
#     c_print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
#     # output = trainer.predict(model, dataloader)
#     output = trainer.predict(model, dataloader)
#
#     c_print('Finished performing inference.')
#     logits = [output[i].logits for i in range(len(output))]
#     logits = torch.cat(logits)
#     predictions = torch.softmax(logits, dim=1)
#
#     return predictions
#

# def get_predictions_manual(model, dm, config, mode):
#     """
#     This function performs inference using a trained model in Pytorch Lightning
#     :param trainer:
#     :param model:
#     :param dm:
#     :param config:
#     :return: list of predictions
#     """
#
#     # TODO make sure that dropout is still enabled for MC-based strategies (unsure whether this is disabled for train.predict)
#     # TODO make sure that all strategies using dataloaders only use NON SHUFFLED dataloaders to preserve meaningful order of examples and indices
#
#     dataloader = dm.unlabelled_dataloader()
#     c_print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
#     model.cuda()
#
#     if mode == 'least-confidence':
#
#         predictions = []
#         for i, batch in enumerate(dataloader):
#
#             batch = {key: value.cuda() for key, value in batch.items()}
#             prediction = model.active_step(batch, i)
#             predictions.extend(prediction)
#
#             # TODO delete this when done!
#             if i % 100 == 0 and i > 0:
#                 break
#
#         return predictions
#
#     if mode == 'entropy':
#
#         result = []
#         for i, batch in enumerate(dataloader):
#
#             batch = {key: value.cuda() for key, value in batch.items()}
#             predictions = model.active_step(batch, i)
#             result.extend(predictions)
#
#             # TODO delete this when done!
#             if i % 100 == 0 and i > 0:
#                 break
#
#         return result
#
#     if mode == 'mc':
#
#         result = []
#         for i, batch in enumerate(dataloader):
#
#             batch = {key: value.cuda() for key, value in batch.items()}
#             predictions = model.mc_step(batch, i)
#
#             result.extend(predictions)
#
#             # TODO delete this when done!
#             if i % 100 == 0 and i > 0:
#                 break
#
#         return result
#
#     if mode == 'bald':
#
#         informations = []
#         for i, batch in enumerate(dataloader):
#
#             batch = {key: value.cuda() for key, value in batch.items()}
#             information = model.mc_step(batch, i)
#
#             informations.extend(information)
#
#             # TODO delete this when done!
#             if i % 100 == 0 and i > 0:
#                 break
#
#         return informations


def select_acquisition_fn(fn_id):
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
        acquisition_fn = Coreset()

    elif fn_id == 'badge':
        raise NotImplementedError

    elif fn_id == 'alps':
        raise NotImplementedError

    elif fn_id == 'contrastive':
        raise NotImplementedError

    elif fn_id == 'cartography':
        raise NotImplementedError

    elif fn_id == 'dal':
        raise NotImplementedError

    else:
        raise KeyError('No acquisition function found for {}'.format(fn_id))

    return acquisition_fn
