from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import RobertaForMaskedLM
from PIL import Image
from torch.utils.data import DataLoader

import os
import torch
import torchmetrics
import torchvision
import wandb

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from dataset.cub_caption import CapCUB200
from util import WarmupLinearSchedule
from model.roberta_cls import RobertaClassificationHead


class LitRobBERTaClassification(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        roberta = RobertaForMaskedLM.from_pretrained(config.pretrained_path)
        self.embedding = roberta.roberta
        self.cls_head = RobertaClassificationHead(config.cls_head)

        self.init_dataset()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()


    def training_step(self, batch, batch_idx):
        inputs, masks, targets = batch

        outputs = self.cls_head(self.embedding(inputs, attention_mask=masks)[0])

        loss = F.cross_entropy(outputs.view(-1, self.config.num_classes), targets.view(-1))
        train_acc = self.train_accuracy(torch.argmax(outputs, dim=-1), targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True,
                sync_dist=True)

        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute(),
                prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs, masks, targets = batch
        outputs = self.cls_head(self.embedding(inputs, attention_mask=masks)[0])

        loss = F.cross_entropy(outputs.view(-1, self.config.num_classes), targets.view(-1))
        val_acc = self.val_accuracy(torch.argmax(outputs, dim=-1), targets)

        self.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.val_accuracy.compute(),
                prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, masks, targets = batch
        outputs = self.cls_head(self.embedding(inputs, attention_mask=masks)[0])

        loss = F.cross_entropy(outputs.view(-1, self.config.num_classes), targets.view(-1))
        test_acc = self.test_accuracy(torch.argmax(outputs, dim=-1), targets)

        self.log("test_acc", test_acc, on_step=False, on_epoch=True, logger=True,
                sync_dist=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True,
                sync_dist=True)

        return loss

    def test_epoch_end(self, outs):
        test_acc = self.test_accuracy.compute()
        self.log("test_acc_epoch", test_acc, logger=True, sync_dist=True)

    def configure_optimizers(self):
        if self.config.warmup:
            optimizer = torch.optim.SGD([
                {"params": self.embedding.parameters()},
                {"params": self.cls_head.parameters()},
            ], lr=self.config.lr, momentum=self.config.momentum)
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=500, t_total=374*self.config.epoch)
            return (
                [optimizer],
                [scheduler]
            )
        else:
            return torch.optim.SGD([
                {"params": self.embedding.parameters()},
                {"params": self.cls_head.parameters()},
            ], lr=self.config.lr, momentum=self.config.momentum)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.config.batch_size,
                        shuffle=True, pin_memory=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                        pin_memory=True, num_workers=self.config.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.config.batch_size,
                        pin_memory=True, num_workers=self.config.num_workers)

    def init_dataset(self):

        self.train_set = CapCUB200(root=self.config.root, tokenizer_path=self.config.pretrained_path, train=True)
        self.test_set = CapCUB200(root=self.config.root, tokenizer_path=self.config.pretrained_path, train=False)
