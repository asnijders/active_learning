"""
This Python script contains implementations for:
- loading and combining various NLI datasets into a single dataset
- Pytorch Lightning Dataset and DataModule classes
"""

import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_dataset(input_dir, dataset_id, split):
    """
    This function takes a dataset id and reads the corresponding .json split in an appropriate Pandas DataFrame
    :param input_dir:
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
        data_path = '{}/snli_1.0/snli_1.0_{}.jsonl'.format(input_dir, split)
        dataset = pd.read_json(data_path, lines=True)
        dataset = dataset[['sentence1', 'sentence2', 'gold_label', 'pairID']]
        dataset = dataset.drop(dataset[dataset.gold_label.str.contains('-')].index)  # drop examples with no gold label

    elif dataset_id == 'ANLI':

        anli_dfs = []  # TODO think about how I want to implement logic for multiple rounds
        for data_round in ['R1', 'R2', 'R3']:
            data_path = '{}/anli_v1.0/{}/{}.jsonl'.format(input_dir, data_round, split)
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
    print(f"{'{} {} size:'.format(dataset_id, split):<30}{len(dataset):<32}", flush=True)

    return dataset


def combine_datasets(input_dir, datasets, split):
    """
    This function takes a list of NLI dataset names and
    concatenates all examples from each corresponding dataset
    for the provided data split (train/dev/test) into a single multi-dataset.

    :param input_dir:
    :param datasets: list with dataset names
    :param split: string indicating data split of interest
    :return: DataFrame with examples and labels for all datasets for split of interest
    """

    # If we only consider a single dataset, we can just read and return it
    if len(datasets) == 1:
        return read_dataset(input_dir, datasets[0], split)

    # If we consider multiple datasets we have to combine them into a single dataset

    # 1. create empty dataframe to store examples from all datasets
    dataset_list = []

    # 2. load individual datasets and append to list
    for dataset_id in datasets:
        dataset = read_dataset(input_dir, dataset_id, split)

        # 3. add dataset to multi-dataset
        dataset_list.append(dataset)

    # 4. combine individual datasets into single dataset
    combined_dataset = pd.concat(dataset_list, axis=0)
    # 5. possibly shuffle examples (?)

    # reset index
    combined_dataset = combined_dataset.reset_index(drop=True)
    combined_dataset.ID = combined_dataset.index

    print(f"{'Total {} size:'.format(split):<30}{len(combined_dataset):<32}", '\n', flush=True)

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

    def __init__(self, config, split):
        """
        :param datasets: datasets: list with dataset names
        :param split: string indicating which split should be accessed
        :param model: string indicating which language model will be used for tokenization
        """

        # create single multi-dataset for desired data split (train, dev or test)
        # for the training split, we consider all examples as unlabelled and start training with a seed set L

        # TODO: add a unique ID to each example in the unlabelled pool,
        #  _just_ after creating the full dataframes. Then we can use that as an extra check for duplicates

        self.input_dir = config.input_dir
        self.datasets = config.datasets
        self.seed_size = config.seed_size
        self.max_length = config.max_length
        self.model_id = config.model_id

        if split == 'train':

            # first, we combine multiple NLI datasets into a single dataset and compile them in unlabeled pool U
            self.U = combine_datasets(self.input_dir, self.datasets, split)

            # TODO add a downsampling mechanism here
            # if self.downsample is True:

            self.L = self.label_instances_randomly(k=self.seed_size)

        else:

            self.L = combine_datasets(self.input_dir, self.datasets, split)

        self.data = self.L
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
        #
        # # tokenize sentence and convert to sequence of ids
        # tokenized_input_seq_pair = self.tokenizer.encode_plus(text=premise,
        #                                                       text_pair=hypothesis,
        #                                                       add_special_tokens=True,
        #                                                       max_length=350,
        #                                                       padding=False,
        #                                                       return_attention_mask=True,
        #                                                       return_token_type_ids=True,
        #                                                       return_tensors='pt',
        #                                                       truncation=True)
        #
        # input_ids = tokenized_input_seq_pair['input_ids'].squeeze(0)
        # token_type_ids = tokenized_input_seq_pair['token_type_ids'].squeeze(0)
        # attention_masks = tokenized_input_seq_pair['attention_mask'].squeeze(0)
        #
        # sample = {'input_ids': input_ids,
        #           'token_type_ids': token_type_ids,
        #           'attention_masks': attention_masks,
        #           'label': label}

        sample = {'premise': premise,
                  'hypothesis': hypothesis,
                  'label': label}

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

    def set_k(self, k):
        """
        if k is an integer we draw k samples from the unlabelled pool
        if k is a percentage we draw the corresponding % from the unlabelled pool
        """

        if k > 1:
            pass
        elif 0 < k < 1:
            k = int(k * len(self.U))
        else:
            raise ValueError('value for k must be percentage or integer > 1')

        return k

    def label_instances_randomly(self, k):
        """
        This function randomly selects k examples from the unlabelled set U
        and transfers them to the labelled set L
        :param k: size of initial pool of labelled examples
        :return:
        """

        # select k instances from U to be labelled for initial seed L.
        # Make sure to remove these instances from U.
        k = self.set_k(k)
        self.U = self.U.reset_index(drop=True)

        print('Drawing {} random samples from unlabelled set U for seed set L..'.format(k), flush=True)

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
        print('Labelled {} new instances'.format(len(new_examples)), flush=True)
        print(f"{'Total size unlabelled pool:':<30}{len(self.U):<32}", flush=True)
        print(f"{'Total size labelled pool:':<30}{len(self.L):<32}", flush=True)

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

    def __init__(self, config):
        super().__init__()

        # self.input_dir = config.input_dir
        # self.datasets = config.datasets
        # self.seed_size = config.seed_size
        # self.max_length = config.max_length
        # self.batch_size = config.batch_size
        # self.num_workers = config.num_workers

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer,
                                                padding='longest',
                                                max_length=config.max_length)

        self.train = None
        self.val = None
        self.test = None
        self.label2id = {"entailment": 0,
                         "contradiction": 1,
                         "neutral": 2}

    def batch_tokenize(self, batch):

        premises = [sample['premise'] for sample in batch]
        hypotheses = [sample['hypothesis'] for sample in batch]
        labels = [sample['label'] for sample in batch]

        labels = torch.tensor(labels)

        # tokenize sentence and convert to sequence of ids
        tokenized_input_seq_pairs = self.tokenizer.__call__(text=premises,
                                                            text_pair=hypotheses,
                                                            add_special_tokens=True,
                                                            max_length=350,
                                                            padding='longest',
                                                            return_attention_mask=True,
                                                            return_token_type_ids=True,
                                                            return_tensors='pt',
                                                            truncation=True)

        input_ids = tokenized_input_seq_pairs['input_ids']
        token_type_ids = tokenized_input_seq_pairs['token_type_ids']
        attention_masks = tokenized_input_seq_pairs['attention_mask']

        padded_batch = {'input_ids': input_ids,
                        'token_type_ids': token_type_ids,
                        'attention_masks': attention_masks,
                        'labels': labels}

        # TODO verify that batches are indeed of variable seq length

        return padded_batch

    def setup(self, stage=None):

        if stage == 'fit':

            print('\nBuilding train pool..', flush=True)
            if self.train is None:
                self.train = DataPool(config=self.config,
                                      split='train', )

            print('\nBuilding dev and test sets..', flush=True)
            if self.val is None:
                self.val = DataPool(config=self.config,
                                    split='dev')
            if self.test is None:
                self.test = DataPool(config=self.config,
                                     split='test')

        if stage == 'test':
            pass

        print('Done building datasets!\nFitting model on initial seed dataset L..', flush=True)

    def train_dataloader(self):

        return DataLoader(self.train,
                          collate_fn=self.batch_tokenize,
                          shuffle=True,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers)

    def val_dataloader(self):

        return DataLoader(self.val,
                          collate_fn=self.batch_tokenize,
                          shuffle=False,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers)

    def test_dataloader(self):

        return DataLoader(self.test,
                          shuffle=False,
                          collate_fn=self.batch_tokenize,
                          batch_size=self.config.batch_size,
                          num_workers=self.config.num_workers)
