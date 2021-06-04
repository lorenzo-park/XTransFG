from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

import hydra
import os
import torch
import pytorch_lightning as pl

from pl_model.vit import LitViT
from pl_model.xfg_concat import LitXFGConcat
from pl_model.xfg_cross import LitXFGCrossAttn
from pl_model.xfg_cross_rec import LitXFGCrossAttnRec
from pl_model.xfg_concat_backbone import LitXFGConcatWithBackbone
from pl_model.xfg_cross_backbone import LitXFGCrossWithBackbone
from pl_model.xfg_cross_dr import LitXFGCrossAttnDR
from pl_model.xfg_concat_dr import LitXFGConcatDR
from pl_model.xfg_concat_encoded_dr import LitXFGConcatEncodedDR
from pl_model.resnet import LitResNet


def get_model(config):
    if config.model == "vit":
        return LitViT(config)
    elif config.model == "xfg_cross":
        return LitXFGCrossAttn(config)
    elif config.model == "xfg_cross_dr":
        return LitXFGCrossAttnDR(config)
    elif config.model == "xfg_concat_dr":
        return LitXFGConcatDR(config)
    elif config.model == "xfg_concat_encoded_dr":
        return LitXFGConcatEncodedDR(config)
    elif config.model == "xfg_cross_rec":
        return LitXFGCrossAttnRec(config)
    elif config.model == "xfg_concat":
        return LitXFGConcat(config)
    elif config.model == "xfg_cross_backbone":
        return LitXFGCrossWithBackbone(config)
    elif config.model == "xfg_concat_backbone":
        return LitXFGConcatWithBackbone(config)
    elif config.model == "resnet":
        return LitResNet(config)


@hydra.main(config_name="config")
def run(config):
    if config.logger:
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project="xfg",
            name=config.model,
            config=config,
        )
    else:
        logger = pl.loggers.TestTubeLogger(
            "output", name=f"vit")
        logger.log_hyperparams(config)

    pl.seed_everything(config.seed)

    early_stop_callback = EarlyStopping(
        monitor='val_acc_epoch',
        min_delta=0.00,
        patience=config.patience,
        verbose=False,
        mode='max'
    )
    if config.gpus > 1:
        trainer = pl.Trainer(
            callbacks=[early_stop_callback],
            precision=16,
            deterministic=True,
            check_val_every_n_epoch=1,
            gpus=config.gpus,
            logger=logger,
            max_epochs=config.epoch,
            weights_summary="top",
            accelerator='ddp',
            # plugins=DDPPlugin(find_unused_parameters=False),
        )
    else:
        trainer = pl.Trainer(
            callbacks=[early_stop_callback],
            precision=16,
            deterministic=True,
            check_val_every_n_epoch=1,
            gpus=config.gpus,
            logger=logger,
            max_epochs=config.epoch,
            weights_summary="top",
        )

    model = get_model(config)
    trainer.fit(model)
    trainer.test()

    if config.model == "vit":
        torch.save(model.model.state_dict(), os.path.join(config.save_path, "vit_cub.pt"))

if __name__ == '__main__':
  run()