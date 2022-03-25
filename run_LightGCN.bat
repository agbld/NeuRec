python ./esun/set_config_LightGCN.py 2019-06-30 --lr 0.0025 --reg 0.000125 --embed_size 128 --n_layers 4 --batch_size 1024 --epochs 200 --n_fold 100 --adj_type pre
python main_esun.py

python ./esun/set_config_LightGCN.py 2019-06-30 --lr 0.0025 --reg 0.0000625 --embed_size 128 --n_layers 4 --batch_size 1024 --epochs 200 --n_fold 100 --adj_type pre
python main_esun.py