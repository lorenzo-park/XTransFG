from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from torch.utils.data import DataLoader

import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from dataset.cub import CUB200
from util import WarmupLinearSchedule
from pl_model.xfg import LitXFGCrossAttn
from xfg_cross_config import config


train_transform=transforms.Compose([
    transforms.Resize((600, 600), InterpolationMode.BILINEAR),
    transforms.RandomCrop((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_set = CUB200(root="./data", train=True, caption=True, transform=train_transform)

if config.logger:
    from pytorch_lightning.loggers import WandbLogger
    logger = WandbLogger(
        project="xfg",
        name=f"vit"
    )
else:
    logger = pl.loggers.TestTubeLogger(
        "output", name=f"vit")
    logger.log_hyperparams(config)

pl.seed_everything(config.seed)
trainer = pl.Trainer(
    precision=16,
    deterministic=True,
    check_val_every_n_epoch=1,
    gpus=config.gpus,
    logger=logger,
    max_epochs=config.epoch,
    weights_summary="top",
    accelerator='ddp',
)

model = LitXFGCrossAttn(config)
trainer.fit(model)
trainer.test()
