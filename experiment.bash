python run.py model=vit
python run.py model=xfg_cross
python run.py model=xfg_concat
python run.py transformer.shared_cls=true model=xfg_concat

python run.py transformer.num_layers_fusion=4 model=xfg_concat
python run.py transformer.num_layers_fusion=8 model=xfg_concat
python run.py transformer.num_layers_fusion=12 model=xfg_concat
python run.py transformer.num_layers_fusion=16 model=xfg_concat
