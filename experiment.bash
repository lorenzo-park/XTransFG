# python run.py model=vit gpus=1

# python run.py model=xfg_cross encoder.transformer.num_layers=1 decoder.transformer.num_layers=1
# python run.py model=xfg_cross_backbone encoder.transformer.num_layers=1 decoder.transformer.num_layers=1
# python run.py model=xfg_concat transformer.num_layers_fusion=1
# python run.py model=xfg_concat_backbone transformer.num_layers_fusion=1

# python run.py model=xfg_cross encoder.transformer.num_layers=4 decoder.transformer.num_layers=4
# python run.py model=xfg_cross_backbone encoder.transformer.num_layers=4 decoder.transformer.num_layers=4
# python run.py model=xfg_concat transformer.num_layers_fusion=4
# python run.py model=xfg_concat_backbone transformer.num_layers_fusion=4

# python run.py model=xfg_cross encoder.transformer.num_layers=6 decoder.transformer.num_layers=6
# python run.py model=xfg_cross_backbone encoder.transformer.num_layers=6 decoder.transformer.num_layers=6
# python run.py model=xfg_concat transformer.num_layers_fusion=6
# python run.py model=xfg_concat_backbone transformer.num_layers_fusion=6

# python run.py model=xfg_cross encoder.transformer.num_layers=12 decoder.transformer.num_layers=4
# python run.py model=xfg_concat transformer.num_layers_fusion=8
# python run.py model=xfg_concat transformer.num_layers_fusion=12
# python run.py model=xfg_cross encoder.transformer.num_layers=3 decoder.transformer.num_layers=3
# python run.py model=xfg_cross encoder.transformer.num_layers=5 decoder.transformer.num_layers=5

# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1
# python run.py model=xfg_concat_dr transformer.num_layers_fusion=1


# python run.py model=xfg_cross_dr encoder.transformer.num_layers=2 decoder.transformer.num_layers=2
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=2 decoder.transformer.num_layers=2

# python run.py model=xfg_cross_dr encoder.transformer.num_layers=3 decoder.transformer.num_layers=3
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=3 decoder.transformer.num_layers=3

# python run.py model=xfg_cross_dr encoder.transformer.num_layers=4 decoder.transformer.num_layers=4
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=4 decoder.transformer.num_layers=4

# python run.py model=xfg_cross_dr encoder.transformer.num_layers=4 decoder.transformer.num_layers=6
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=4 decoder.transformer.num_layers=6

# python run.py model=xfg_cross_dr encoder.transformer.num_layers=8 decoder.transformer.num_layers=8
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=8 decoder.transformer.num_layers=8


python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg"


python run.py model=vit gpus=1 seed=None project="xfg-final"
python run.py model=vit gpus=1 seed=None project="xfg-final"
python run.py model=vit gpus=1 seed=None project="xfg-final"
python run.py model=vit gpus=1 seed=None project="xfg-final"
python run.py model=vit gpus=1 seed=None project="xfg-final"

python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"

python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 seed=None project="xfg-final"