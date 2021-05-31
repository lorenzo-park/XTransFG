from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pytorch_lightning as pl

from pl_model.xfg import LitXFGConcat
from xfg_concat_config import config


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
)

model = LitXFGConcat(config)
trainer.fit(model)
trainer.test()