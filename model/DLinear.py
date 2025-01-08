#!/usr/bin/env python
# coding: utf-8

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import silhouette_samples, silhouette_score

from setup import IndexSetup, ColumnsSetup

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, conf, data, mode='train'):
                
        self.conf = conf
        self.index_setup = IndexSetup(conf, data)
        self.columns_setup = ColumnsSetup(conf, data)
        
        if mode == 'train':
            self.base_index_set = self.index_setup.base_index_set_train
        else:
            self.base_index_set = self.index_setup.base_index_set_test
        
        self.use_var_size = len(self.columns_setup.use_var_names)
        self.time_var_size = len(self.columns_setup.time_var_names)
        self.value_var_size = len(self.columns_setup.value_var_names)
        self.prev_times = self.index_setup.prev_times
        self.next_times = self.index_setup.next_times
        
        self.values = torch.FloatTensor(data[self.columns_setup.value_var_names].values).to(DEVICE)
        self.times = torch.FloatTensor(data[self.columns_setup.time_var_names].values).to(DEVICE)
        self.indexs = torch.IntTensor(data.index).to(DEVICE)
                            
    def __len__(self):
        
        return len(self.base_index_set)
        
    def __getitem__(self, key):
        
        base_index = self.base_index_set[key]
        prev_index, next_index = self.index_setup.get_indexes(base_index)
        
        x_t = self.times[prev_index, :]
        x_v = self.values[prev_index, :]
        y = self.values[next_index, :]
        pos = self.indexs[next_index]
        
        return x_v, x_t, y, pos

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class PowerModel(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, ds):
        super(PowerModel, self).__init__()
        # self.seq_len = configs.seq_len
        # self.pred_len = configs.pred_len
        self.seq_len = 192
        self.pred_len = 24

        # Decompsition Kernel Size
        kernel_size = 11
        self.decompsition = series_decomp(kernel_size)
        # self.individual = configs.individual 
        # self.channels = configs.enc_in 
        self.individual = False # 채널별 독립 학습 여부 -> STLF 이기 때문에 False
        
        x_v, x_t, y, pos = ds[0]
        self.v_size = x_v.shape[1]
        self.channels = self.a_size = x_v.shape[1] + x_t.shape[1] # Encoder Input Size

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len)).to(DEVICE)
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len)).to(DEVICE)

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len).to(DEVICE)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len).to(DEVICE)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        self.linear = nn.Sequential(nn.Linear(self.a_size, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 1),
                                  ).to(DEVICE)

    def forward(self, x_v, x_t):
        x_v_sum = x_v[:, -24:] # 최근 24시간의 전력량 합을 최종 예측값에 곱할 것인지

        x_v = (1 + x_v).log()
        x = torch.cat((x_v, x_t), dim=2)
        
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 최종 결과 계산
        x = seasonal_output + trend_output
        # x[:,:self.v_size,:] *= x_v_sum.permute(0,2,1)
        x = x.permute(0,2,1)
        x = self.linear(x)
        # x = x * x_v_sum
        
        return x

def training(conf, data, epoch_size=30, batch_size=16, lr=0.001, shuffle=True):
    
    ds_train = CustomDataset(conf, data, mode='train')
    ds_test = CustomDataset(conf, data, mode='test')
    
    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle=shuffle)
    dl_test = DataLoader(ds_test, batch_size = batch_size)

    model = PowerModel(ds_train)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epoch_count = batch_count = 0
    
    ress = []
    err_best = np.inf
    model_best = model
        
    for epoch in range(epoch_size):

        epoch_count += 1
                
        for x_v, x_t, y, pos in dl_train:
            
            batch_count += 1
            model.train()
            model.zero_grad()
        
            output = model(x_v, x_t)
                        
            true = y.sum(dim=2).flatten()
            pred = output.flatten()
            loss = loss_function(pred, true)
            err_train = loss.tolist()
            loss.backward()
            optimizer.step()

            model.eval()

            if not batch_count % 10:
                
                with torch.no_grad():

                    rets = []
                    for x_v, x_t, y, pos in dl_test:

                        output = model(x_v, x_t)

                        ret = pd.DataFrame({'epoch':epoch_count, 
                                            'batch':batch_count, 
                                            'pos': pos.flatten().tolist(), 
                                            'act_val': y.sum(dim=2).flatten().tolist(), 
                                            'pred_val': output.flatten().tolist()})
                        rets.append(ret)

                res = pd.concat(rets).reset_index(drop=True)
                res['err_train'] = err_train
                err_test = np.mean(np.abs(res['act_val']-res['pred_val'])/res['act_val'])

                prf = f'(Epoch : {epoch_count}, Batch : {batch_count}) '
                prf += f'Train Loss(MSE) {err_train:.5f} / Test Loss(MAPE): {err_test*100:.2f}'
                prf += ' '*10
                print(prf, end='\r')

                if err_test < err_best:
                    err_best = err_test
                    model_best = model

                ress.append(res)
            
    perf = pd.concat(ress).reset_index(drop=True)
    
    return perf, model_best