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
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
import transformers
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from scipy.stats import entropy
from torchmetrics.functional import accuracy


class TransformerModel(LightningModule):
    """
    This class implements a Lightning Module for several Transformer-based models.
    """
    def __init__(self, dropout, lr, model_id, batch_size):
        super().__init__()

        # transformers.logging.set_verbosity_error()
        self.dropout = dropout # dropout applied to BERT
        self.lr = lr # learning rate
        self.max_length = 180
        self.batch_size = batch_size

        # self.accuracy = metrics.Accuracy() # for logging to lightning
        self.tokenizer = AutoTokenizer.from_pretrained(model_id) #TODO replace str with model arg!
        # load pre-trained, uncased, sequence-classification BERT model
        # TODO make sure the correct from_pretrained model is being loaded

        self.encoder = BertForSequenceClassification.from_pretrained(model_id, # TODO replace str with model arg
                                                                     num_labels=3,
                                                                     hidden_dropout_prob=self.dropout)

        # logging
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def deconstruct(self):
        self.encoder = None

    def configure_optimizers(self):
        "This method handles optimization of params for PyTorch lightning"
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
        acc = accuracy(preds, labels)
        metrics = {'train_acc': acc, 'loss': loss}
        self.log_dict(metrics, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def validation_step(self, batch, batch_idx):

        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def test_step(self, batch, batch_idx):

        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics

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
        acc = accuracy(preds, labels)
        return loss, acc

    def active_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        output = self(input_ids=input_ids,
                      attention_masks=attention_masks,
                      token_type_ids=token_type_ids,
                      labels=labels)

        prediction = torch.softmax(output.logits.detach(), dim=1).cpu().numpy()

        return prediction

    def mc_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        mc_average = []

        for k in range(10):

            output = self(input_ids=input_ids,
                          attention_masks=attention_masks,
                          token_type_ids=token_type_ids,
                          labels=labels)

            prediction = torch.softmax(output.logits.detach(), dim=1).cpu().numpy()
            mc_average.append(prediction)

        return np.mean(np.asarray(mc_average), axis=0)

    def bald_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        predictions = []
        disagreements = []

        for k in range(10):

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
        return list(entropies - disagreements)

    def embedding_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        output = self(input_ids=input_ids,
                      attention_masks=attention_masks,
                      token_type_ids=token_type_ids,
                      labels=labels)

        embedding = output.hidden_states[-1][:, 0, :].detach().unsqueeze(1).cpu().numpy()

        return embedding
