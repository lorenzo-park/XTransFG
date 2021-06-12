from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

import torch
import torchmetrics
import torchvision
import wandb

import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from dataset.cub import CUB200
from model.xfg import XFGConcatEncodedDR
from utils.lr_schedule import WarmupCosineSchedule
from utils.autoaug import AutoAugImageNetPolicy


class LitXFGConcatEncodedDR(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = XFGConcatEncodedDR(config)
        self.model.load_from(np.load(config.pretrained_dir))
        self.config = config

        self.init_dataset()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        self.plot = True

    def training_step(self, batch, _):
        inputs, txt, targets = batch

        outputs, _ = self.model(inputs, txt.squeeze(1))
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
        inputs, txt, targets = batch
        outputs, attn_weights = self.model(inputs, txt.squeeze(1))

        loss = F.cross_entropy(outputs.view(-1, self.config.num_classes), targets.view(-1))
        val_acc = self.val_accuracy(torch.argmax(outputs, dim=-1), targets)

        self.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.plot:
            images = attn_weights[-1][0].squeeze(0).unsqueeze(1).repeat(1, 3, 1, 1)
            self.attn_weights = torchvision.utils.make_grid(images)
            self.plot = False
        return loss

    def validation_epoch_end(self, outs):
        self.logger.experiment.log({"attn_weights": [wandb.Image(self.attn_weights)]})
        self.plot = True
        self.log("val_acc_epoch", self.val_accuracy.compute(),
                prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, txt, targets = batch

        outputs, attn_weights = self.model(inputs, txt.squeeze(1))

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
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum, weight_decay=1e-5)
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=374*self.config.epoch)
            return (
                [optimizer],
                [scheduler]
            )
        else:
            return torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum, weight_decay=1e-5)

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

        self.train_set = CUB200(root=self.config.root, train=True, caption=True, transform=train_transform)
        self.test_set = CUB200(root=self.config.root, train=False, caption=True, transform=test_transform)