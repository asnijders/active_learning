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

# local imports
from src.coresets import CoresetGreedy


class AcquisitionFunction:
    def __init__(self):

        pass

    def acquire_instances(self, config, model, dm, k, trainer):
        pass


class RandomSampler(AcquisitionFunction):
    def __init__(self):
        super().__init__()

    def acquire_instances(self, config, model, dm, k, trainer):

        k_random_indices = np.random.choice(a=len(dm.train.U),
                                            size=int(k),
                                            replace=False)

        return k_random_indices


class LeastConfidence(AcquisitionFunction):

    def __init__(self):
        super().__init__()

    def acquire_instances(self, config, model, dm, k, trainer):
        """
        This function implements least-confidence acquisition
        """

        # predictions = get_predictions(model=model,
        #                               dm=dm,
        #                               config=config,
        #                               trainer=trainer)

        with torch.no_grad():
            dataloader = dm.unlabelled_dataloader()
            print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
            model.cuda()

            predictions = []
            for i, batch in enumerate(tqdm(dataloader)):

                batch = {k: v.cuda() for k, v in batch.items()}
                prediction = model.active_step(batch, i)
                predictions.extend(prediction)

            predictions = np.asarray(predictions)
            max_probabilities = np.max(predictions, axis=1)

            probability_gap = 1 - np.array(max_probabilities)
            least_confident_indices = np.argsort(probability_gap)[:k][::-1]

        return least_confident_indices

        # # obtain desired proportion
        # # k = dm.train.set_k(k)
        #
        # # keep track of most probable label probabilities
        # max_probabilities = []
        #
        # dataloader = DataLoader(dataset=dm.train,
        #                         collate_fn=dm.batch_tokenize,
        #                         batch_size=config.batch_size,
        #                         shuffle=False,
        #                         num_workers=config.num_workers)
        #
        #
        #
        # with torch.no_grad():
        #
        #     # disable gradient tracking
        #     model.eval()
        #
        #     # loop over unlabelled data and store probabilities
        #     for i, batch in enumerate(dataloader):
        #
        #         outputs = model.active_step(batch=batch,
        #                                     batch_idx=i)
        #         logits = outputs.logits
        #         probabilities, _ = torch.max(torch.softmax(logits, dim=1), dim=1)
        #         probabilities = probabilities.detach().cpu().numpy()
        #         probabilities = list(probabilities)
        #         max_probabilities.extend(probabilities)
        #
        #
        #     # select the k examples with the largest gap between 1 and the most probable label probability
        #     # TODO fix this function
        #     probability_gap = 1 - np.array(max_probabilities)
        #     least_confident_indices = np.argsort(probability_gap)[:k][::-1]
        #
        #     # re-enable grad computation
        #     model.train()
        #
        # return least_confident_indices


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
            print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
            model.cuda()

            if self.mode == 'entropy':

                entropies = []
                for i, batch in enumerate(dataloader):

                    batch = {k: v.cuda() for k, v in batch.items()}
                    prediction = model.active_step(batch, i)
                    entropies.append(entropy(prediction, axis=1))

                max_entropy_indices = np.argsort(entropies)[:k][::-1]

                return max_entropy_indices

            elif self.mode == 'mc-entropy':

                mc_entropies = []
                for i, batch in enumerate(dataloader):
                    batch = {k: v.cuda() for k, v in batch.items()}
                    mc_entropy = model.mc_step(batch, i)
                    mc_entropies.append(mc_entropy)

                mc_entropy_indices = np.argsort(mc_entropies)[:k][::-1]

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
            print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
            model.cuda()

            informations = []
            for i, batch in enumerate(dataloader):

                batch = {k: v.cuda() for k, v in batch.items()}
                information = model.bald_step(batch, i)
                informations.extend(information)

            bald_indices = np.argsort(informations)[:k][::-1]

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
                    batch = {k: v.cuda() for k, v in batch.items()}
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
#     print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
#     # output = trainer.predict(model, dataloader)
#     output = trainer.predict(model, dataloader)
#
#     print('Finished performing inference.')
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
#     print('Performing inference on unlabelled data for {}..'.format(config.acquisition_fn))
#     model.cuda()
#
#     if mode == 'least-confidence':
#
#         predictions = []
#         for i, batch in enumerate(dataloader):
#
#             batch = {k: v.cuda() for k, v in batch.items()}
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
#             batch = {k: v.cuda() for k, v in batch.items()}
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
#             batch = {k: v.cuda() for k, v in batch.items()}
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
#             batch = {k: v.cuda() for k, v in batch.items()}
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
        acquisition_fn = MaxEntropy(mode='entropy')

    elif fn_id == 'mc-max-entropy':
        acquisition_fn = MaxEntropy(mode='mc-entropy')

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
