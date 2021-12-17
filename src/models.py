"""
This Python script implements a wrapper class for
a variety of language models (RoBERTa, DistilBERT, ELECTRA)
"""

# Global modules
import argparse
import sys
import time
import os

# PyTorch modules
import torch
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
import transformers
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

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

    def tokenize_batch(self, batch):

        premise = batch['premise']
        hypothesis = batch['hypothesis']

        tokenized_input_seq_pair = self.tokenizer.__call__(text=premise,
                                                           text_pair=hypothesis,
                                                           max_length=self.max_length,
                                                           padding='longest',
                                                           return_token_type_ids=True,
                                                           truncation=True,
                                                           return_attention_mask=True,
                                                           return_tensors="pt")

        input_ids = tokenized_input_seq_pair['input_ids'].squeeze(1)
        token_type_ids = tokenized_input_seq_pair['token_type_ids'].squeeze(1)
        attention_masks = tokenized_input_seq_pair['attention_mask'].squeeze(1)

        return input_ids, token_type_ids, attention_masks

    def active_step(self, batch, batch_idx):
        """
        step function for running the model on unlabelled data
        :param batch:
        :param batch_idx:
        :return:
        """

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_masks = batch['attention_masks']
        labels = batch['labels']

        outputs = self(input_ids=input_ids,
                       attention_masks=attention_masks,
                       token_type_ids=token_type_ids,
                       labels=labels)

        return outputs

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
