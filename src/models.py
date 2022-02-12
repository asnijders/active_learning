"""
This Python script implements a wrapper class for
a variety of language models (RoBERTa, DistilBERT, ELECTRA)
"""

# Global modules
import argparse
import sys
import time
import os
import numpy as np

# PyTorch modules
import torch
from torch import nn
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
import transformers
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification, AutoConfig
from scipy.stats import entropy
from torchmetrics.functional import accuracy
import requests
import time


from pytorch_lightning.utilities import rank_zero_only


def get_model(model_id, dropout):
    """
    Loads model and tokenizer in try-except block in case of server denials
    """

    num_labels = 3
    model_config = AutoConfig.from_pretrained(model_id, num_labels=num_labels, dropout=dropout)

    done = False
    while done is False:

        try:
            if model_id == 'bert-base-uncased':
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id, config=model_config)

                done = True

            elif model_id == 'roberta-base':
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id, config=model_config)

                done = True

            elif model_id == 'bert-large-uncased':
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id, config=model_config)

                done = True

            else:
                raise KeyError('No model found for {}'.format(model_id))

        except requests.HTTPError as exception:
            print('Got internal server error. Trying to download model again in 10 seconds..', flush=True)
            print(exception)
            time.sleep(10)

    return tokenizer, model


class TransformerModel(LightningModule):
    """
    This class implements a Lightning Module for several Transformer-based models.
    """
    def __init__(self,
                 dropout,
                 lr,
                 model_id,
                 batch_size,
                 acquisition_fn,
                 mc_iterations,
                 num_gpus,
                 separate_test_sets):

        super().__init__()
        self.save_hyperparameters()

        # transformers.logging.set_verbosity_error()
        self.dropout = dropout # dropout applied to BERT
        self.lr = lr  # learning rate
        self.max_length = 180
        self.batch_size = batch_size
        self.acquisition_fn = acquisition_fn
        self.mc_iterations = mc_iterations
        self.predictions = None
        self.num_gpus = num_gpus
        self.separate_test_sets = separate_test_sets
        self.test_set_id = ''
        self.dev_set_id = ''

        # load pre-trained, uncased, sequence-classification BERT model
        self.tokenizer, self.encoder = get_model(model_id=model_id,
                                                 dropout=dropout)

        # init metrics
        self.init_metrics()

    def init_metrics(self):
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def deconstruct(self):
        self.encoder = None

    def configure_optimizers(self):
        """This method handles optimization of params for PyTorch lightning"""
        return Adam(self.parameters(), lr=self.lr)

    def forward(self, input_ids, attention_masks, token_type_ids, labels):

        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_masks,
                              token_type_ids=token_type_ids,
                              labels=labels,
                              output_hidden_states=True)

        return output

    def training_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        outputs = self(input_ids=input_ids,
                       attention_masks=attention_masks,
                       token_type_ids=token_type_ids,
                       labels=labels)

        loss = outputs.loss
        preds = outputs.logits
        acc = self.train_acc(preds, labels)
        metrics = {'train_acc': acc, 'loss': loss}
        self.log_dict(metrics,
                      batch_size=self.batch_size,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True,
                      sync_dist=False)
        return metrics

    def validation_step(self, batch, batch_idx):

        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {'{}val_acc'.format(self.dev_set_id): acc,
                   '{}val_loss'.format(self.dev_set_id): loss}
        self.log_dict(metrics,
                      batch_size=self.batch_size,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True,
                      sync_dist=False)

        return metrics

    def validation_end(self, outputs):

        avg_loss = torch.stack([x['{}val_loss'.format(self.dev_set_id)] for x in outputs]).mean()
        avg_acc = torch.stack([x['{}val_acc'.format(self.dev_set_id)] for x in outputs]).mean()
        return {
            '{}val_loss'.format(self.dev_set_id): avg_loss,
            '{}val_acc'.format(self.dev_set_id): avg_acc,
            'progress_bar': {'{}val_loss'.format(self.dev_set_id): avg_loss,
                             '{}val_acc'.format(self.dev_set_id): avg_acc}}

    def test_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        outputs = self(input_ids=input_ids,
                       attention_masks=attention_masks,
                       token_type_ids=token_type_ids,
                       labels=labels)

        loss = outputs.loss
        preds = outputs.logits
        acc = self.test_acc(preds, labels)

        # loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"{}test_acc".format(self.test_set_id): acc,
                   "{}test_loss".format(self.test_set_id): loss}

        self.log_dict(metrics,
                      batch_size=self.batch_size,
                      on_step=True,
                      on_epoch=True,
                      prog_bar=True,
                      logger=True,
                      sync_dist=False)
        return metrics

    # def test_end(self, outputs):
    #
    #     avg_loss = torch.stack([x['{}test_loss'.format(self.test_set_id)] for x in outputs]).mean()
    #     avg_acc = torch.stack([x['{}test_acc'.format(self.test_set_id)] for x in outputs]).mean()
    #     return {
    #         '{}test_loss'.format(self.test_set_id): avg_loss,
    #         '{}test_acc'.format(self.test_set_id): avg_acc,
    #         'progress_bar': {'{}test_loss'.format(self.test_set_id): avg_loss,
    #                          '{}test_acc'.format(self.test_set_id): avg_acc}}

    def _shared_eval_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        outputs = self(input_ids=input_ids,
                       attention_masks=attention_masks,
                       token_type_ids=token_type_ids,
                       labels=labels)

        loss = outputs.loss
        preds = outputs.logits
        acc = self.val_acc(preds, labels)
        return loss, acc

    def predict_step(self, batch, batch_idx):

        if self.acquisition_fn == 'random':
            return self.active_step(batch, batch_idx)

        elif self.acquisition_fn == 'least-confidence':
            return self.active_step(batch, batch_idx)

        elif self.acquisition_fn == 'max-entropy':
            return self.active_step(batch, batch_idx)

        elif self.acquisition_fn == 'mc-max-entropy':
            return self.mc_step(batch, batch_idx)

        elif self.acquisition_fn == 'bald':
            return self.bald_step(batch, batch_idx)

        elif self.acquisition_fn == 'coreset':
            return self.embedding_step(batch, batch_idx)

    def on_predict_epoch_end(self, results):

        """
        The predictions are currently in a list of shape [num_batches x world_size x batch_size x num_classes]
        The problem is that each gathered tensor contains a batch belonging to the first, second and third GPU.
        But how do I know in what order I should put these different tensors to ensure that the order is the same
        as the way they were fed?

        One thing I can try is to run one experiment on 3 GPUs, and print all the predictions.
        Then run the same experiment on a single GPU and print those predictions.
        And then see what order they are in, and whether this is consistent.

        Thankfully, it appears that across runs with the same seeds, the logits are the same.

        To verify:
        - are the logits the same irrespective of whether I run on a single GPU, or on 3 GPUs?
        - what order are the different GPU tensors in?
        - what is a good way of re-ordering the per-gpu tensors?
        """
        # since DDP divides the data among different GPUs, we need to 're-unite' their predictions after inference
        # self.all_gather gathers the predictions from separate gpu processes.
        # >>> list of length num_batches where each elem is a tensor of Size ([world_size, batch_size, num_classes])
        # print('finished inference, awaiting gather')

        if self.num_gpus == 1:
            return None

        # print(predictions)
        predictions = self.all_gather(data=results)[0]

        # iterate over list of multi-gpu tensors
        ordered_predictions = []
        for multi_gpu_tensor in predictions:
            # split multi-gpu tensor into list of single-gpu tensors
            split_tensors = torch.tensor_split(multi_gpu_tensor, self.num_gpus, dim=0)  # TODO replace 3 with a config gpu arg
            # remove implicit first dimension
            split_tensors = [tensor.squeeze(0) for tensor in split_tensors]
            # concatenate single-gpu tensors along batch dimension and append to list
            ordered_predictions.append(torch.cat(split_tensors, dim=0))

        # concatenate list of tensors along batch dimension once more
        predictions = torch.cat(ordered_predictions, dim=0)
        print(predictions.size()) # should return torch.Size([<num_batches x world_size x batch_size>, 3])
        # >>> torch.Size([17532, 3])

        # if predictions.size()

        # print('finished gather operation')
        self.predictions = predictions.cpu()

        # print('\n returned predictions: \n')
        # print(predictions)

        return None

    def active_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        output = self(input_ids=input_ids,
                      attention_masks=attention_masks,
                      token_type_ids=token_type_ids,
                      labels=labels)

        prediction = torch.softmax(output.logits.detach(), dim=1)

        return prediction

    def apply_dropout(self, m):
        """simple fn for enabling all dropout modules in model"""
        if type(m) == nn.Dropout:
            m.train()

    def mc_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        self.encoder.apply(self.apply_dropout)

        output = torch.vstack([

            torch.softmax(self(input_ids=input_ids,
                               attention_masks=attention_masks,
                               token_type_ids=token_type_ids,
                               labels=labels).logits,
                          dim=1).unsqueeze(0) for _ in range(self.mc_iterations)]).mean(dim=0)

        return output

        # for k in range(10):
        #
        #     output = self(input_ids=input_ids,
        #                   attention_masks=attention_masks,
        #                   token_type_ids=token_type_ids,
        #                   labels=labels)
        #
        #     prediction = torch.softmax(output.logits.detach(), dim=1).cpu().numpy()
        #     mc_average.append(prediction)
        #
        # print(mc_average)
        #
        # # mc_average is an object of shape k (MC iters) x batch_size x num_classes
        # # our interest is in the mean over the different models, so we average over k, i.e. dim 0
        # # we return an array of shape batch_size x num_classes
        # return np.mean(np.asarray(mc_average), axis=0)

    def bald_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        # re-enable dropout for MC dropout
        self.encoder.apply(self.apply_dropout)

        # def get_outputs(_input_ids,
        #                 _attention_masks,
        #                 _token_type_ids,
        #                 _labels):
        #
        #     single_output = torch.softmax(self(input_ids=_input_ids,
        #                                        attention_masks=_attention_masks,
        #                                        token_type_ids=_token_type_ids,
        #                                        labels=_labels).logits, dim=1)
        #
        #     single_prediction = torch.distributions.distribution.Distribution(single_output)
        #     single_disagreements = single_prediction.entropy()
        #
        #     return single_prediction.numpy(), single_disagreements.numpy()
        #
        # for _ in range(self.mc_iterations):
        #
        #     prediction, disagreement = get_outputs(_input_ids=input_ids,
        #                                            _attention_masks=attention_masks,
        #                                            _token_type_ids=token_type_ids,
        #                                            _labels=labels)
        #     predictions.append(prediction)
        #     disagreements.append(disagreement)
        #
        # entropies = entropy(np.mean(predictions, axis=0), axis=1)
        # disagreements = np.mean(disagreements, axis=0)
        # return list(entropies - disagreements)

        # mc_output = [get_outputs(_input_ids=input_ids,
        #                          _attention_masks=attention_masks,
        #                          _token_type_ids=token_type_ids,
        #                          _labels=labels) for _ in range(self.mc_iterations)]
        #
        # # output = [get_outputs(batch) for _ in range(self.mc_iterations)]
        #
        # output = torch.vstack([
        #
        #     torch.softmax(self(input_ids=input_ids,
        #                        attention_masks=attention_masks,
        #                        token_type_ids=token_type_ids,
        #                        labels=labels).logits,
        #                   dim=1).unsqueeze(0) for _ in range(self.mc_iterations)]).mean(dim=0)

        predictions = []
        disagreements = []

        for _ in range(self.mc_iterations):

            output = self(input_ids=input_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          labels=labels)

            prediction = torch.softmax(output.logits.detach(), dim=1).cpu().numpy()

            predictions.append(prediction)
            disagreements.append(entropy(prediction, axis=1))

        # Compute Entropy of Average
        entropies = entropy(np.mean(predictions, axis=0), axis=1)
        disagreements = np.mean(disagreements, axis=0)
        return torch.tensor(entropies - disagreements)

    def embedding_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        output = self(input_ids=input_ids,
                      attention_masks=attention_masks,
                      token_type_ids=token_type_ids,
                      labels=labels)

        embedding = output.hidden_states[-1][:, 0, :].detach().unsqueeze(1).cpu()

        return embedding

