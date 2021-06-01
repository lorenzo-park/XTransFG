from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

import hydra
import pytorch_lightning as pl

from pl_model.vit import LitViT
from pl_model.xfg import LitXFGConcat, LitXFGCrossAttn


def get_model(config):
    if config.model == "vit":
        return LitViT(config)
    elif config.model == "xfg_cross":
        return LitXFGCrossAttn(config)
    elif config.model == "xfg_concat":
        return LitXFGConcat(config)


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
        monitor='val_loss',
        min_delta=0.00,
        patience=config.patience,
        verbose=False,
        mode='min'
    )

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
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    model = get_model(config)
    trainer.fit(model)
    trainer.test()

if __name__ == '__main__':
  run()