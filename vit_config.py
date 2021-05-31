from omegaconf import DictConfig


config = DictConfig({
    "patch_size": 32,
    "split": "overlap",
    "slide_step": 24,
    "hidden_size": 768,
    "dropout": 0.1,
    "max_len": 100,
    "classifier": "token",
    "transformer": {
        "mlp_dim": 3072,
        "num_heads": 12,
        "num_layers": 12,
        "attention_dropout_rate": 0.0,
    },
    "num_classes": 200,
    "batch_size": 16,
    "num_workers": 8,
    "image_size": 448,
    "lr": 3e-2,
    "seed": 42,
    "momentum": 0.9,
    "epoch": 10000,
    "gpus": 2,
    "logger": True,
    "patience": 10,
    "pretrained_dir": "./pretrained/vit/imagenet21k_ViT-B_32.npz",
})