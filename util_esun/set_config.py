import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
dim = args.dim
lr = args.lr

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

with open("./util_esun/NGCF.properties", "w") as f:
    f.write(ngcf_prop)
