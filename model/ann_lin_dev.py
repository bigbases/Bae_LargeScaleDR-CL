#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from setup import ColumnsSetup, IndexSetup

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


class PowerModel(nn.Module):
    
    def __init__(self, ds):
        super().__init__()
        
        x_v, x_t, y, pos = ds[0]
        self.v_size = x_v.shape[1]
        self.a_size = x_v.shape[1] + x_t.shape[1]
        
        self.lin1 = nn.Sequential(nn.Linear(self.a_size, 256),
                                  nn.BatchNorm1d(192),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 24),
                                  nn.BatchNorm1d(192),
                                  nn.LeakyReLU(),
                                  ).to(DEVICE)
        
        self.lin2 = nn.Sequential(nn.Linear(192*24, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 24),
                                #   nn.LeakyReLU(),
                                  ).to(DEVICE)
        

    def forward(self, x_v, x_t):
        
        b_size = x_v.shape[0]
        x_v_sum = x_v.sum(dim=2, )[:,-24:]
        x_v = (1+x_v).log()
        x = torch.cat((x_v, x_t), dim=2)
        
        output = self.lin1(x)
        output = output.reshape(b_size, -1)
        output = self.lin2(output)
        output *= x_v_sum
                        
        return output


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
            model.zero_grad()   # 각 배치마다 기울기 초기화: 단점이 있을 것 같다
        
            output = model(x_v, x_t)
                        
            loss = loss_function(output.flatten(), y.sum(dim=2).flatten())
            err_train = loss.tolist()
            loss.backward()
            optimizer.step()

            if not batch_count % 10:
                
                model.eval()
                
                with torch.no_grad():   # 평가를 위해 기울기 학습을 멈춤

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
                
                model.train()
            
    perf = pd.concat(ress).reset_index(drop=True)
    
    return perf, model_best
