python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 proj_dim=5
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 proj_dim=20
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 proj_dim=49 # Linguistic token length
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 proj_dim=100
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 proj_dim=300
python run.py model=xfg_cross_rec encoder.transformer.num_layers=1 decoder.transformer.num_layers=1 project="xfg-final" seed=44 proj_dim=785 # Visual token length
rm -rf outputs