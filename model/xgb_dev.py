#!/usr/bin/env python
# coding: utf-8

import os, sys, time, re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.manifold import TSNE
from umap import UMAP
from xgboost import XGBRegressor
from setup import IndexSetup, ColumnsSetup

BASE_MODEL_CODE = 'xgb_dev'


# In[7]:


# SEASON = 'summer'
# REDUCE_METHOD = 'only'
# N_CLUSTER = 1
SEASON = sys.argv[1]
REDUCE_METHOD = sys.argv[2]
N_CLUSTER = int(sys.argv[3])


# In[8]:


MODEL_CODE = f"""{REDUCE_METHOD}_nc{N_CLUSTER}_{BASE_MODEL_CODE}_{SEASON}"""


# In[9]:


SAVE_FOLD = Path() / 'result' / MODEL_CODE
SAVE_FOLD.mkdir(parents=True, exist_ok=True)


# In[ ]:


assert not (SAVE_FOLD / 'error.bin').exists(), 'Already Done.'


# In[10]:


conf_summer = {'season':'summer', 'prev_times':192, 'next_times':24, 
               'test_ratio':0.2, 'train_sample_ratio':1, 'summer_months':[4, 5, 6, 7, 8, 9],
               'time_flag_var':'time', 'time_var_format':'.+_d[1-7]_t_$', 'value_var_format':'^h[0-9]+'}

conf_winter = {'season':'winter', 'prev_times':192, 'next_times':24, 
               'test_ratio':0.2, 'train_sample_ratio':1, 'summer_months':[4, 5, 6, 7, 8, 9],
               'time_flag_var':'time', 'time_var_format':'.+_d[1-7]_t_$', 'value_var_format':'^h[0-9]+'}

if SEASON == 'summer':
    CONF = conf_summer
elif SEASON == 'winter':
    CONF = conf_winter
else:
    raise ValueError(f'Invaild Season name. ({SEASON})')


# In[11]:



data = pd.read_pickle('../ireland_power_1h.bin')
data = data[data.columns[~data.columns.str.contains('^minute_')]].reset_index(drop=True)


# In[12]:


TIME_SET = {}


# In[13]:


INDEX_SETUP = IndexSetup(CONF, data)
COLUMNS_SETUP = ColumnsSetup(CONF, data)   


# In[14]:


# Reduce Dimension 
# Log transform
if REDUCE_METHOD == 'tsne':
    model_reduce = TSNE()
elif REDUCE_METHOD == 'umap':
    model_reduce = UMAP()
else:
    model_reduce = ''

arr_v = data.loc[INDEX_SETUP.train_period_index, COLUMNS_SETUP.value_var_names].values.T
arr_v = np.log(arr_v + 1)

time_start = time.time()
if model_reduce != '':
    arr_v = model_reduce.fit_transform(arr_v)  
time_end = time.time()
TIME_SET['REDUCE_DIM'] = time_end - time_start


# In[15]:


# K-means
model_clust = KMeans(n_clusters=N_CLUSTER, random_state=0)
time_start = time.time()
clust_id_set = model_clust.fit_predict(arr_v)
time_end = time.time()
TIME_SET['CLUSTERING'] = time_end - time_start

# Save Clustering Set
pd.to_pickle(clust_id_set, SAVE_FOLD / 'clust_id_set.bin')


# In[16]:



info_clust = pd.DataFrame({'var_value':COLUMNS_SETUP.value_var_names, 'clust_id':clust_id_set})


# In[25]:


perfs = [] 
models = {}

start_time = time.time()
for clust_id in range(N_CLUSTER):
    
    var_value_new = info_clust.loc[info_clust['clust_id'] == clust_id, 'var_value'].to_list()
    
    var_use = COLUMNS_SETUP.base_var_names 
    var_use = var_use.append(COLUMNS_SETUP.time_var_names) 
    var_use = var_use.append(pd.Index(var_value_new))    


    data_new = data.loc[:, var_use]
    index_setup = IndexSetup(CONF, data_new)
    columns_setup = ColumnsSetup(CONF, data_new)
    y_col_index = list(range(index_setup.next_times))


    ys = []
    y_sum = data_new.loc[:,columns_setup.value_var_names].sum(axis=1)    
    for next_time in range(index_setup.next_times):
        next_index = index_setup.base_index_set_train + (next_time + 1)
        y = y_sum[next_index]
        y.index = index_setup.base_index_set_train
        ys.append(y)
        
    Y = pd.concat(ys, axis=1)
    X = data_new.loc[index_setup.base_index_set_train, columns_setup.use_var_names].values

    ymodels = {}
    

    for idx in y_col_index:
        print(time.ctime(), f'Cluster ID : {clust_id} / Next Time : {idx}')
        y = Y[idx].values
        model = XGBRegressor()
        model = model.fit(X, y)
        ymodels[idx] = model    


    nX = data_new.loc[index_setup.base_index_set_test, columns_setup.use_var_names].values


    rets = []
    for idx in y_col_index:
        model = ymodels[idx]
        ret = pd.Series(model.predict(nX), index=index_setup.base_index_set_test+idx+1)
        rets.append(ret)
        

    perf = pd.concat(rets).sort_index()
    perf = pd.DataFrame({'pred_val':perf})


    perf['act_val'] = data_new.loc[perf.index, columns_setup.value_var_names].sum(axis=1)
    perf = data_new.loc[:,columns_setup.base_var_names].join(perf).dropna().reset_index()
    perf['clust_id'] = clust_id

    perfs.append(perf)
    models[clust_id] = ymodels
            
end_time = time.time()

TIME_SET['LEARNING'] = end_time - start_time

result = pd.concat(perfs).reset_index(drop=True)
info_test = result.groupby('time')[['act_val', 'pred_val']].sum().reset_index()
info_test['ae'] = (info_test['act_val'] - info_test['pred_val']).abs() 
info_test['se']= (info_test['act_val'] - info_test['pred_val']).pow(2)
info_test['ar'] = info_test['act_val'].abs()
info_test['sr'] = info_test['act_val'].pow(2)
info_test['ape'] = info_test['ae'] / info_test['act_val']

test_mape = info_test['ape'].mean() * 100
test_rmse = np.power(info_test['se'].mean(), .5)
test_nmae = info_test['ae'].mean() / info_test['ar'].mean()
test_nrmse = np.power(info_test['se'].mean(), .5) / np.power(info_test['sr'].mean(), .5)
err_info = pd.DataFrame({'mape':test_mape, 'rmse':test_rmse, 
                         'nmae':test_nmae, 'nrmse':test_nrmse}, index=[0])

err_info['season'] = SEASON

pd.to_pickle(err_info, SAVE_FOLD / 'error.bin')
pd.to_pickle(TIME_SET, SAVE_FOLD / 'time_set.bin')
pd.to_pickle(perfs, SAVE_FOLD / 'performance.bin')
pd.to_pickle(models, SAVE_FOLD / 'models.bin')

