import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path', type=str)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

dataset_path = args.dataset_path
epochs = args.epochs
batch_size = args.batch_size
dim = args.dim
lr = args.lr

# data preprocessing
if os.path.exists('./esun/dataset/' + dataset_path + '/interaction_train.rating') == False:
    df = pd.read_csv('./esun/dataset/' + dataset_path + '/interaction_train.csv', usecols=['cust_no', 'wm_prod_code', 'txn_dt'])
    df['txn_dt'] = pd.to_datetime(df['txn_dt'])
    df['txn_dt'] = df.txn_dt.values.astype(np.int64) // 10 ** 9
    df.to_csv('./esun/dataset/' + dataset_path + '/interaction_train.rating', index=False, header=False, sep='\t')

# update config
neurec_prop = '''[default]

######## model
recommender=NGCF
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

ngcf_prop = '''[hyperparameters]
epochs={epochs}
batch_size={batch_size}
embedding_size={dim}
layer_size=[{dim},{dim}]
learning_rate={lr}
node_dropout_flag=False
adj_type=norm
alg_type=ngcf
loss_function=BPR
learner=adam
reg=0.0
node_dropout_ratio=0.1
mess_dropout_ratio=0.1
#tnormal: truncated_normal_initializer, uniform: random_uniform_initializer,
#normal: random_normal_initializer, xavier_normal, xavier_uniform, 
#he_normal, he_uniform. Defualt: tnormal
embed_init_method=xavier_normal
weight_init_method=xavier_normal
stddev=0.01
verbose=1'''.format(epochs=epochs, batch_size=batch_size, dim=dim, lr=lr)

with open("./esun/conf/NGCF.properties", "w") as f:
    f.write(ngcf_prop)
