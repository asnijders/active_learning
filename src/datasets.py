"""
This Python script contains implementations for:
- loading and combining various NLI datasets into a single dataset
- Pytorch Lightning Dataset and DataModule classes
"""

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_dataset(dataset_id, split):
    """
    This function takes a dataset id and reads the corresponding .json split in an appropriate Pandas DataFrame
    :param dataset_id: str indicating which dataset should be read
    :param split: str indicating which split should be read
    :return: DataFrame with examples (Premise, Hypothesis, Label, ID)
    """

    # TODO implement the above logic for MNLI, FEVER, ETC!

    def replace_labels(label):
        label_conversions = {'e': 'entailment',
                             'c': 'contradiction',
                             'n': 'neutral'}

        return label_conversions[label]

    if dataset_id == 'SNLI':

        # TODO only consider snli with gold labels?
        data_path = 'resources/data/snli_1.0/snli_1.0_{}.jsonl'.format(split)
        dataset = pd.read_json(data_path, lines=True)
        dataset = dataset[['sentence1', 'sentence2', 'gold_label', 'pairID']]

    elif dataset_id == 'ANLI':

        anli_dfs = []  # TODO think about how I want to implement logic for multiple rounds
        for data_round in ['R1', 'R2', 'R3']:
            data_path = 'resources/data/anli_v1.0/{}/{}.jsonl'.format(data_round, split)
            anli_dataset = pd.read_json(data_path, lines=True)
            anli_dataset = anli_dataset[['context', 'hypothesis', 'label', 'uid']]  # get rid of unnecessary columns
            anli_dataset['label'] = anli_dataset['label'].apply(replace_labels)  # ensures consistently named labels
            anli_dfs.append(anli_dataset)

        dataset = pd.concat(anli_dfs, axis=0)

    # elif dataset_id == 'FEVER':
    #
    #     raise NotImplementedError
    #
    #     # TODO figure out equivalence of FEVER labels w.r.t. other NLI data?
    #     data_path = 'resources/data/nli_fever/{}_fitems.jsonl'.format(split)
    #     dataset = pd.read_json(data_path, lines=True)
    #     dataset = dataset[['context', 'query', 'label', 'fid']]

    else:
        raise KeyError('No dataset found for "{}"'.format(dataset_id))

    # ensure consistent headers per dataset DataFrame
    dataset.columns = ['Premise', 'Hypothesis', 'Label', 'ID']
    print(f"{'{} {} size:'.format(dataset_id, split):<30}{len(dataset):<32}")

    return dataset


def combine_datasets(datasets, split):
    """
    This function takes a list of NLI dataset names and
    concatenates all examples from each corresponding dataset
    for the provided data split (train/dev/test) into a single multi-dataset.

    :param datasets: list with dataset names
    :param split: string indicating data split of interest
    :return: DataFrame with examples and labels for all datasets for split of interest
    """

    # If we only consider a single dataset, we can just read and return it
    if len(datasets) == 1:
        return read_dataset(datasets[0], split)

    # If we consider multiple datasets we have to combine them into a single dataset

    # 1. create empty dataframe to store examples from all datasets
    dataset_list = []

    # 2. load individual datasets and append to list
    for dataset_id in datasets:
        dataset = read_dataset(dataset_id, split)

        # 3. add dataset to multi-dataset
        dataset_list.append(dataset)

    # 4. combine individual datasets into single dataset
    combined_dataset = pd.concat(dataset_list, axis=0)
    # 5. possibly shuffle examples (?)

    # reset index
    combined_dataset = combined_dataset.reset_index(drop=True)
    combined_dataset.ID = combined_dataset.index

    print(f"{'Total {} size:'.format(split):<30}{len(combined_dataset):<32}", '\n')

    return combined_dataset


class EvaluationSet(Dataset):
    """
    This class implements the Pytorch Lightning Dataset object for separate NLI evaluation sets
    """

    def __init__(self, dataset_id, split, max_length=180, model='bert-base-uncased'):
        """
        :param dataset_id: str indicating which dataset should be read
        :param split: string indicating which split (dev/test) should be read
        """
        # create single multi-dataset for desired data split (train, dev or test)
        # for the training split, we consider all examples as unlabelled and start training with a seed set L
        self.text = read_dataset(dataset_id, split)

        # initialise tokenizer
        self.max_length = max_length
        self.label2id = {"entailment": 0,
                         "contradiction": 1,
                         "neutral": 2}

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        premise, hypothesis, label, _ = self.text.iloc[idx]
        label = self.label2id[label]

        sample = {"premise": premise,
                  "hypothesis": hypothesis,
                  "label": label}

        return sample


class DataPool(Dataset):
    """
    This class implements the Pytorch Lightning Dataset object for multiple NLI datasets
    """

    def __init__(self, datasets, seed_size, split, max_length=180, model='bert-base-uncased'):
        """
        :param datasets: datasets: list with dataset names
        :param split: string indicating which split should be accessed
        :param model: string indicating which language model will be used for tokenization
        """

        # create single multi-dataset for desired data split (train, dev or test)
        # for the training split, we consider all examples as unlabelled and start training with a seed set L

        # TODO: add a unique ID to each example in the unlabelled pool,
        #  _just_ after creating the full dataframes. Then we can use that as an extra check for duplicates

        if split == 'train':

            # first, we combine multiple NLI datasets into a single dataset and compile them in unlabeled pool U
            self.U = combine_datasets(datasets, split)

            if self.downsample is True:


            # we then randomly draw k instances from U for the initial set L
            # if seed_size is an integer, an equivalent number of examples is drawn for L
            # if seed size is a float, an equivalent percentage of examples from U is drawn from L
            if seed_size > 1:
                self.seed_size = self.seed_size
            elif 0 < seed_size < 1:
                self.seed_size = self.set_seed_size(seed_size)

            self.L = self.label_instances_randomly(k=self.seed_size)

        else:

            self.L = combine_datasets(datasets, split)

        self.data = self.L
        self.max_length = max_length
        self.label2id = {"entailment": 0,
                         "contradiction": 1,
                         "neutral": 2}

    def __len__(self):
        return len(self.L)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        premise, hypothesis, label, _ = self.data.iloc[idx]
        label = self.label2id[label]

        sample = {"premise": premise,
                  "hypothesis": hypothesis,
                  "label": label}

        return sample

    def set_mode(self, mode):
        """
        Calling this function sets self.data to point to:
        - the labelled set, if we seek to access the labelled examples in L
        - the unlabelled set, if we seek to access the unlabelled examples U
        :param mode:
        :return:
        """

        if mode == 'L':
            self.data = self.L

        elif mode == 'U':
            self.data = self.U

        else:
            raise KeyError('{} is not a valid mode. Use mode=L or mode=U.'.format(mode))

    def set_seed_size(self, percentage):
        assert 0 < percentage < 1
        num_unlabelled_examples = len(self.U)
        return int(percentage * num_unlabelled_examples)

    def label_instances_randomly(self, k):
        """
        This function randomly selects k examples from the unlabelled set U
        and transfers them to the labelled set L
        :param k: size of initial pool of labelled examples
        :return:
        """
        # select k instances from U to be labelled for initial seed L.
        # Make sure to remove these instances from U.
        self.U = self.U.reset_index(drop=True)

        print('Drawing {} random samples from unlabelled set U for seed set L..'.format(k))

        # initialize empty seed dataset L
        L = pd.DataFrame(columns=self.U.columns)

        # select k instances from U and move them to L
        random_indices = list(np.random.randint(low=0,
                                                high=len(self.U),
                                                size=k))

        labelled_examples = self.U.iloc[random_indices]
        L = L.append(labelled_examples).reset_index(drop=True)
        self.U = self.U.drop(labelled_examples.index).reset_index(drop=True)
        print(f"{'Total size unlabelled pool:':<30}{len(self.U):<32}")
        print(f"{'Total size labelled pool:':<30}{len(L):<32}")

        return L

    def label_instances(self, indices):
        """
        This function takes an array of indices and transfers the corresponding examples from the unlabelled pool U
        to the labelled pool L
        :param indices: np array with indices of samples that we want to label based on some acquisition function
        :return:
        """

        # TODO: can I come up with a nice mechanism for making sure that no mistakes are made when moving samples
        # TODO: from the labeled to unlabeled sets?

        new_examples = self.U.iloc[indices]  # Take examples from unlabelled pool
        self.L = self.L.append(new_examples).reset_index(drop=True)  # Add them to the labelled pool
        self.U = self.U.drop(new_examples.index).reset_index(drop=True)  # Remove examples from unlabelled pool
        self.assert_disjoint()  # check whether U and L are disjoint
        print('Labelled {} new instances'.format(len(new_examples)))
        print(f"{'Total size unlabelled pool:':<30}{len(self.U):<32}")
        print(f"{'Total size labelled pool:':<30}{len(self.L):<32}")

        return None

    def assert_disjoint(self):
        """
        re-usable function for ensuring that U and L do not overlap
        """
        unlabelled_indices = set(self.U.ID.tolist())
        labelled_indices = set(self.L.ID.tolist())
        assert unlabelled_indices.isdisjoint(labelled_indices)
        return None


class GenericDataModule(pl.LightningDataModule):
    """
    This Lightning module produces DataLoaders using DataPool instances
    """

    def __init__(self, datasets, seed_size, max_length, batch_size, num_workers):
        super().__init__()

        self.datasets = datasets
        self.seed_size = seed_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage=None):

        if stage == 'fit':

            print('\nBuilding train pool..')
            if self.train is None:
                self.train = DataPool(datasets=self.datasets,
                                      seed_size=self.seed_size,
                                      split='train',
                                      max_length=self.max_length)

            print('\nBuilding dev and test sets..')
            if self.val is None:
                self.val = DataPool(datasets=self.datasets,
                                    seed_size=self.seed_size,
                                    split='dev',
                                    max_length=self.max_length)
            if self.test is None:
                self.test = DataPool(datasets=self.datasets,
                                     seed_size=self.seed_size,
                                     split='test',
                                     max_length=self.max_length)

        if stage == 'test':
            pass

        print('Done building datasets!\nFitting model on initial seed dataset L..')

    def train_dataloader(self):

        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):

        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)
