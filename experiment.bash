# Run 0 for ablation study
python run.py model=vit project="xfg-final" seed=42
python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42
python run.py model=roberta_cls project="xfg-final" seed=42 lr=1e-3

python run.py model=vit_roberta_cls project="xfg-final" seed=42
python run.py model=xfg_concat_encoded_dr encoder.transformer.num_layers=1 project="xfg-final" seed=42
rm -rf outputs

# Run 1 for ablation study
python run.py model=vit project="xfg-final" seed=43
python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43
python run.py model=roberta_cls project="xfg-final" seed=43 lr=1e-3

python run.py model=vit_roberta_cls project="xfg-final" seed=43
python run.py model=xfg_concat_encoded_dr encoder.transformer.num_layers=1 project="xfg-final" seed=43
rm -rf outputs

# Run 2 for ablation study
python run.py model=vit project="xfg-final" seed=44
python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44
python run.py model=roberta_cls project="xfg-final" seed=44 lr=1e-3

python run.py model=vit_roberta_cls project="xfg-final" seed=44
python run.py model=xfg_concat_encoded_dr encoder.transformer.num_layers=1 project="xfg-final" seed=44
rm -rf outputs

# Run 3 for ablation study
python run.py model=vit project="xfg-final" seed=45
python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=45
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=45
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=45
python run.py model=roberta_cls project="xfg-final" seed=45 lr=1e-3

python run.py model=vit_roberta_cls project="xfg-final" seed=45
python run.py model=xfg_concat_encoded_dr encoder.transformer.num_layers=1 project="xfg-final" seed=45
rm -rf outputs

# Run 4 for ablation study
python run.py model=vit project="xfg-final" seed=46
python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=46
python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=46
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=46
python run.py model=roberta_cls project="xfg-final" seed=46 lr=1e-3

python run.py model=vit_roberta_cls project="xfg-final" seed=46
python run.py model=xfg_concat_encoded_dr encoder.transformer.num_layers=1 project="xfg-final" seed=46
# rm -rf outputs

# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=45
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=46

# # Run 1 for ablation study
# python run.py model=vit project="xfg-final" seed=43 patience=0 epoch=112 warmup=true
# python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43 patience=0 epoch=46
# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43 patience=0 epoch=34
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=43 patience=0 epoch=34
# rm -rf outputs

# # Run 2 for ablation study
# python run.py model=vit project="xfg-final" seed=44 patience=0 epoch=93
# python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 patience=0 epoch=48
# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 patience=0 epoch=34
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 patience=0 epoch=32
# rm -rf outputs

# # Run 3 for ablation study
# python run.py model=vit project="xfg-final" seed=45 patience=0 epoch=90
# python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=45 patience=0 epoch=47
# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=45 patience=0 epoch=30
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=45 patience=0 epoch=32
# rm -rf outputs

# # Run 4 for ablation study
# python run.py model=vit project="xfg-final" seed=46 patience=0 epoch=93
# python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=46 patience=0 epoch=44
# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=46 patience=0 epoch=31
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=46 patience=0 epoch=30
# rm -rf outputs

# # Run 5 for ablation study
# python run.py model=vit project="xfg-final" seed=42 patience=0 epoch=94
# python run.py model=xfg_nocross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42 patience=0 epoch=47
# python run.py model=xfg_cross_dr encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42 patience=0 epoch=33
# python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=42 patience=0 epoch=33