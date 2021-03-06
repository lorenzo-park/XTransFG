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

from dataset.cub_img_cap import ImgCapCUB200
from utils.lr_schedule import WarmupCosineSchedule
from utils.autoaug import AutoAugImageNetPolicy
from model.vit_roberta_cls import ConcatClassificationHead
from model.vit import VisionTransformer
from model.xfg import Encoder, TrainablePositionalEncoding

class LitViTRobBERTa(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = VisionTransformer(config).transformer

        self.img_pos_embedding = TrainablePositionalEncoding(config.visual_token_len+1, config.hidden_size, dropout=config.dropout)
        self.txt_pos_embedding = TrainablePositionalEncoding(config.max_len+1, config.hidden_size, dropout=config.dropout)

        self.img_encoder = Encoder(config.encoder)
        self.txt_encoder = Encoder(config.encoder)

        self.cls_token_img = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cls_token_txt = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.cls_head = ConcatClassificationHead(config.cls_head)

        self.init_dataset()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()


    def training_step(self, batch, batch_idx):
        inputs_imgs, txt, _, _, targets = batch

        cls_token_img = self.cls_token_img.expand(inputs_imgs.shape[0], -1, -1)
        cls_token_txt = self.cls_token_txt.expand(txt.shape[0], -1, -1)

        outputs_imgs, _ = self.model(inputs_imgs)
        outputs_imgs = torch.cat([cls_token_img, outputs_imgs], dim=1)
        outputs_imgs = self.img_pos_embedding(outputs_imgs)
        outputs_imgs, _ = self.img_encoder(outputs_imgs)

        outputs_txts = torch.cat([cls_token_txt, txt.squeeze(1)], dim=1)
        outputs_txts = self.txt_pos_embedding(outputs_txts)
        outputs_txts, _ = self.txt_encoder(outputs_txts)

        outputs = self.cls_head(outputs_imgs, outputs_txts)

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
        inputs_imgs, txt, _, _, targets = batch

        outputs_imgs, _ = self.model(inputs_imgs)
        outputs_imgs, _ = self.img_encoder(outputs_imgs)
        outputs_txts, _ = self.txt_encoder(txt.squeeze(1))
        outputs = self.cls_head(outputs_imgs, outputs_txts)

        loss = F.cross_entropy(outputs.view(-1, self.config.num_classes), targets.view(-1))
        val_acc = self.val_accuracy(torch.argmax(outputs, dim=-1), targets)

        self.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.val_accuracy.compute(),
                prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs_imgs, txt, _, _, targets = batch

        outputs_imgs, _ = self.model(inputs_imgs)
        outputs_imgs, _ = self.img_encoder(outputs_imgs)
        outputs_txts, _ = self.txt_encoder(txt.squeeze(1))
        outputs = self.cls_head(outputs_imgs, outputs_txts)

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
                {"params": self.model.parameters()},
                {"params": self.cls_head.parameters()},
                {"params": self.img_encoder.parameters()},
                {"params": self.txt_encoder.parameters()},
            ], lr=self.config.lr,  momentum=self.config.momentum, weight_decay=1e-5)
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=374*self.config.epoch)
            return (
                [optimizer],
                [scheduler]
            )
        else:
            return torch.optim.SGD([
                {"params": self.model.parameters()},
                {"params": self.cls_head.parameters()},
                {"params": self.img_encoder.parameters()},
                {"params": self.txt_encoder.parameters()},
            ], lr=self.config.lr, momentum=self.config.momentum, weight_decay=1e-5)

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
        train_transform=transforms.Compose([
            transforms.Resize((600, 600), InterpolationMode.BILINEAR),
            transforms.RandomCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            # AutoAugImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform=transforms.Compose([
            transforms.Resize((600, 600), InterpolationMode.BILINEAR),
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.train_set = ImgCapCUB200(root=self.config.root, tokenizer_path=self.config.pretrained_path, transform=train_transform, train=True)
        self.test_set = ImgCapCUB200(root=self.config.root, tokenizer_path=self.config.pretrained_path, transform=test_transform, train=False)
