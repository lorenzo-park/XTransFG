from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

import os
import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl

from dataset.cub import CUB200
from model.resnet import ResNet


class LitResNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        model = ResNet(classes=config.num_classes, pretrained=True)
        self.feature_extractor = model.feature_extractor
        self.classifier = model.classifier
        self.config = config

        self.init_dataset()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        outputs = self.classifier(self.feature_extractor(inputs))
        loss = F.cross_entropy(outputs, targets)
        train_acc = self.train_accuracy(torch.argmax(outputs, dim=-1), targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_acc", train_acc, on_step=False, on_epoch=True,
                sync_dist=True)

        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute(),
                prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        outputs = self.classifier(self.feature_extractor(inputs))

        loss = F.cross_entropy(outputs, targets)
        val_acc = self.val_accuracy(torch.argmax(outputs, dim=-1), targets)

        self.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss


    def validation_epoch_end(self, outs):
        self.log("val_acc_epoch", self.val_accuracy.compute(),
                prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.classifier(self.feature_extractor(inputs))

        loss = F.cross_entropy(outputs, targets)
        test_acc = self.test_accuracy(torch.argmax(outputs, dim=-1), targets)

        self.log("test_acc", test_acc, on_step=False, on_epoch=True, logger=True,
                sync_dist=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True,
                sync_dist=True)

        return loss

    def test_epoch_end(self, outs):
        test_acc = self.test_accuracy.compute()
        self.log("test_acc_epoch", test_acc, logger=True, sync_dist=True)

        torch.save(self.feature_extractor.state_dict(), os.path.join(self.config.save_path, "resnet_cub_feature_extractor.pt"))
        torch.save(self.classifier.state_dict(), os.path.join(self.config.save_path, "resnet_cub_classifier.pt"))

    def configure_optimizers(self):
        model_params = [
            {"params": self.feature_extractor.parameters()},
            {"params": self.classifier.parameters()}
        ]
        optimizer = torch.optim.SGD(model_params, lr=self.config.lr, momentum=self.config.momentum)
        return optimizer

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
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform=transforms.Compose([
            transforms.Resize((600, 600), InterpolationMode.BILINEAR),
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.train_set = CUB200(root=self.config.root, train=True, transform=train_transform)
        self.test_set = CUB200(root=self.config.root, train=False, transform=test_transform)
