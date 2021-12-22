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
from tqdm import tqdm
from scipy.stats import entropy
import copy

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

        # obtain desired proportion
        # k = dm.train.set_k(k)

        # keep track of most probable label probabilities
        max_probabilities = []

        dataloader = DataLoader(dataset=dm.train,
                                collate_fn=dm.batch_tokenize,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers)
        with torch.no_grad():

            # disable gradient tracking
            model.eval()

            # loop over unlabelled data and store probabilities
            for i, batch in enumerate(dataloader):

                outputs = model.active_step(batch=batch,
                                            batch_idx=i)
                logits = outputs.logits
                probabilities, _ = torch.max(torch.softmax(logits, dim=1), dim=1)
                probabilities = probabilities.detach().cpu().numpy()
                probabilities = list(probabilities)
                max_probabilities.extend(probabilities)

                #  TODO remove this once done with debugging
                if i == 0:
                    break

            # select the k examples with the largest gap between 1 and the most probable label probability
            # TODO fix this function
            probability_gap = 1 - np.array(max_probabilities)
            least_confident_indices = np.argsort(probability_gap)[:k][::-1]

            # re-enable grad computation
            model.train()

        return least_confident_indices


class MaxEntropy(AcquisitionFunction):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def acquire_instances(self, config, model, dm, k, dropout_k=10):
        """
        This function implements entropy uncertainty sampling acquisition
        """

        # obtain desired proportion
        # k = dm.train.set_k(k)

        # keep track of most probable label probabilities
        max_entropies = []

        # create dataloader for unlabeled data
        dm.train.set_mode('U')
        dataloader = DataLoader(dataset=dm.train,
                                collate_fn=dm.batch_tokenize,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers)

        if self.mode == 'entropy':

            with torch.no_grad():

                # disable gradient tracking
                model.eval()

                # loop over unlabelled data and store probabilities
                for i, batch in enumerate(dataloader):

                    outputs = model.active_step(batch=batch,  # pass batch through model
                                                batch_idx=i)
                    logits = outputs.logits  # extract logits
                    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()  # obtain prob dist over labs
                    entropies = entropy(probabilities, axis=1)  # compute entropy of prob dist per example
                    entropies = list(entropies)
                    max_entropies.extend(entropies)

                    #  TODO remove this once done with debugging
                    if i == 0:
                        break

                max_entropy_indices = np.argsort(max_entropies)[:k][::-1]

                # re-enable grad computation
                model.train()

            return max_entropy_indices

        elif self.mode == 'mc-entropy':

            with torch.no_grad():

                # loop over unlabelled data and store probabilities
                for i, batch in enumerate(dataloader):

                    batch_probabilities = []
                    for _ in range(dropout_k):  # perform k passes per batch to obtain k MC samples of prob dists

                        outputs = model.active_step(batch=batch,  # pass batch through model
                                                    batch_idx=i)
                        logits = outputs.logits  # extract logits
                        probability = torch.softmax(logits, dim=1).detach().cpu().numpy()  # obtain prob dist over labels
                        batch_probabilities.append(probability)

                    batch_probabilities = np.mean(batch_probabilities, axis=0)  # compute mean probabilities over k samples
                    entropies = entropy(batch_probabilities, axis=1)  # compute entropy over mean prob dists
                    max_entropies.extend(entropies)  # store entropies

                    #  TODO remove this once done with debugging
                    if i == 0:
                        break

            # normalize entropies
            max_entropies /= np.sum(max_entropies)
            mc_max_entropy_indices = np.argsort(max_entropies)[:k][::-1]

            return mc_max_entropy_indices


class BALD(AcquisitionFunction):
    def __init__(self):
        super().__init__()

    def acquire_instances(self, config, model, dm, k, dropout_k=10):
        """
        This function implements entropy uncertainty sampling acquisition
        """

        # obtain desired proportion
        # k = dm.train.set_k(k)

        # keep track of most probable label probabilities
        informations = []

        # create dataloader for unlabeled data
        dataloader = DataLoader(dataset=dm.train,
                                collate_fn=dm.batch_tokenize,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers)

        with torch.no_grad():

            # loop over unlabelled data and store probabilities
            for i, batch in enumerate(dataloader):

                probabilities, disagreement = [], []
                for _ in range(dropout_k):  # perform k passes per batch to obtain k MC samples of prob dists

                    outputs = model.active_step(batch=batch,  # pass batch through model
                                                batch_idx=i)

                    logits = outputs.logits  # extract logits
                    probability = torch.softmax(logits, dim=1).detach().cpu().numpy()  # obtain prob dist over labels
                    probabilities.append(probability)
                    disagreement.append(entropy(probability, axis=1))

                entropies = entropy(np.mean(probabilities, axis=0), axis=1)
                disagreements = np.mean(disagreement, axis=0)
                informations.extend(list(entropies-disagreements))

                #  TODO remove this once done with debugging
                if i == 0:
                    break

        # TODO ensure that we obtain the _top_ k instances (i.e. with maximum mutual info)
        bald_indices = np.argsort(informations)[:k][::-1]

        return bald_indices


class Coreset(AcquisitionFunction):
    # source: https://github.com/svdesai/coreset-al/blob/master/active_learn.py
    def __init__(self):
        super().__init__()

    def acquire_instances(self, config, model, dm, k, dropout_k=10):

        # obtain desired proportion
        # k = dm.train.set_k(k)

        # create separate dataset objects for labeled and unlabeled data
        # set unlabelled dm to unlabelled_dataset
        unlabeled_dataset = dm.train
        # create dataloader for unlabeled data
        unlab_loader = DataLoader(dataset=unlabeled_dataset,
                                  collate_fn=dm.batch_tokenize,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

        # create deepcopy of dm and set mode to train
        labeled_dataset = copy.deepcopy(dm.train)
        labeled_dataset.set_mode('L')
        # create dataloader for labeled data
        lab_loader = DataLoader(dataset=labeled_dataset,
                                collate_fn=dm.batch_tokenize,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers)

        def get_features(encoder, loader):
            """
            static function for performing inference on labeled and unlabeled data
            takes a model and a dataloader, returns a list of embeddings
            """
            features = []
            encoder.eval()

            with torch.no_grad():

                for i, batch in enumerate(loader):

                    output = encoder.active_step(batch=batch,
                                                 batch_idx=i)

                    # get representations in CLS space (following Active Learning for BERT: An Empirical Study)
                    # TODO add an alternative kind of sentence embedding?
                    embeddings = output.hidden_states[-1][:, 0, :].unsqueeze(1).cpu().numpy()
                    features.extend(embeddings)

                    # TODO delete this after done debugging!
                    if i * config.batch_size > k:
                        break

            return features


        # get labeled features
        labeled_features = get_features(encoder=model,
                                        loader=lab_loader)  # (img_name, features)

        # get unlabeled features
        unlabeled_features = get_features(encoder=model,
                                          loader=unlab_loader)  # (img_name, features)

        all_features = labeled_features + unlabeled_features

        labeled_indices = np.arange(0, len(labeled_features))

        coreset = CoresetGreedy(all_features)
        new_batch, max_distance = coreset.sample(already_selected=labeled_indices,
                                                 sample_size=k)

        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        new_batch = [i - len(labeled_features) for i in new_batch]

        return new_batch


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
