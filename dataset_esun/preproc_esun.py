#%%
# import
import pandas as pd
import numpy as np

#%%
# read data
df = pd.read_csv('./2019-06-30/interaction_train.csv', usecols=['cust_no', 'wm_prod_code', 'txn_dt'])

#%%
# preprocess
df['txn_dt'] = pd.to_datetime(df['txn_dt'])
df['txn_dt'] = df.txn_dt.values.astype(np.int64) // 10 ** 9

#%%
# save
df.to_csv('./2019-06-30/interaction_train.rating', index=False, header=False, sep='\t')