# Run 0 for ablation study
# python run.py model=vit project="xfg-final" seed=42
# python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42
# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42
# python run.py model=roberta_cls project="xfg-final" seed=42 lr=1e-3

python run.py model=vit_roberta_cls project="xfg-final" seed=42 lr=1e-3
# python run.py model=xfg_concat_encoded_dr encoder.transformer.num_layers=1 project="xfg-final" seed=42
rm -rf outputs

# Run 1 for ablation study
# python run.py model=vit project="xfg-final" seed=43
# python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43
# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43
# python run.py model=roberta_cls project="xfg-final" seed=43 lr=1e-3

python run.py model=vit_roberta_cls project="xfg-final" seed=43 lr=1d-3
# python run.py model=xfg_concat_encoded_dr encoder.transformer.num_layers=1 project="xfg-final" seed=43
rm -rf outputs

# Run 2 for ablation study
# python run.py model=vit project="xfg-final" seed=44
# python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44
# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44
# python run.py model=roberta_cls project="xfg-final" seed=44 lr=1e-3

python run.py model=vit_roberta_cls project="xfg-final" seed=44 lr=1e-3
# python run.py model=xfg_concat_encoded_dr encoder.transformer.num_layers=1 project="xfg-final" seed=44
rm -rf outputs
