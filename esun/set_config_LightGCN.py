import argparse
import pandas as pd
import os
import numpy as np

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--reg', type=float, default=1e-3)
parser.add_argument('--embed_size', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--n_fold', type=int, default=100)
parser.add_argument('--adj_type', type=str, default='pre')
args = parser.parse_args()

dataset_path = args.dataset_path
lr = args.lr
reg = args.reg
embed_size = args.embed_size
n_layer = args.n_layers
batch_size = args.batch_size
epochs = args.epochs
n_fold = args.n_fold
adj_type = args.adj_type

# data preprocessing
if os.path.exists('./esun/dataset/' + dataset_path + '/interaction_train.rating') == False:
    df = pd.read_csv('./esun/dataset/' + dataset_path + '/interaction_train.csv', usecols=['cust_no', 'wm_prod_code', 'txn_dt'])
    df['txn_dt'] = pd.to_datetime(df['txn_dt'])
    df['txn_dt'] = df.txn_dt.values.astype(np.int64) // 10 ** 9
    df.to_csv('./esun/dataset/' + dataset_path + '/interaction_train.rating', index=False, header=False, sep='\t')

# update config
neurec_prop = '''[default]

######## model
recommender=LightGCN
# model configuration directory
config_dir=./esun/conf

gpu_id=0
gpu_mem=0.8

######## dataset
data.input.path=./esun/dataset/{dataset_path}
data.input.dataset=interaction_train

# data.column.format = UIRT, UIT, UIR, UI
data.column.format=UIT

# separator "\t" " ","::", ","
data.convert.separator='\t'

######## pre-processing/filtering
user_min=0
item_min=0

######## data splitting
# splitter = ratio, loo, given
splitter=ratio
# train set ratio if splitter=ratio
ratio=0.8
by_time=False

######## evaluating
# metric = Precision, Recall, MAP, NDCG, MRR
metric=["Precision", 'Recall']
# topk is int or list of int
topk=[5]
# group_view is list or None, e.g. [10, 20, 30, 40]
group_view=None
rec.evaluate.neg=0
test_batch_size=128
num_thread=8


# data pre-process
# binThold = -1.0 do nothing
# binThold = value, rating > value is changed to 1.0 other is 0.0.
# data.convert.binarize.threshold=0

#will be used to evaluate.'''.format(dataset_path=dataset_path)
with open("./esun/NeuRec.properties", "w") as f:
    f.write(neurec_prop)

lightgcn_prop = '''[hyperparameters]
lr = {lr}
reg = {reg}
embed_size = {embed_size}
n_layers = {n_layers}
batch_size = {batch_size}
epochs = {epochs}
n_fold = {n_fold}
;adj_type = plain, norm, gcmc, pre
adj_type = {adj_type}'''.format(lr=lr, reg=reg, embed_size=embed_size, n_layers=n_layer, batch_size=batch_size, epochs=epochs, n_fold=n_fold, adj_type=adj_type)

with open("./esun/conf/LightGCN.properties", "w") as f:
    f.write(lightgcn_prop)