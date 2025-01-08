#!/usr/bin/env python
# coding: utf-8

import numba
import numpy as np
import pandas as pd

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

class PowerModel(nn.Module):
    def __init__(self, ds):
        super(PowerModel, self).__init__()
        x_v, x_t, y, pos = ds[0]
        self.a_size = int(x_v.shape[1] + x_t.shape[1])
        
        self.use_cuda = True
        self.P = 192
        self.m = self.a_size
        self.hidR = 100
        self.hidC = 100
        self.hidS = 5
        self.Ck = 6
        self.skip = 24
        self.pt = (self.P - self.Ck)//self.skip
        self.hw = 24
        self.output_fun = 'sigmoid'
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m)).to(DEVICE)
        self.GRU1 = nn.GRU(self.hidC, self.hidR).to(DEVICE)
        self.dropout = nn.Dropout(p = 0.2)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS).to(DEVICE)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m).to(DEVICE)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1).to(DEVICE)
        self.output = None
        if (self.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (self.output_fun == 'tanh'):
            self.output = F.tanh
        self.lin = nn.Sequential(nn.Linear(self.a_size, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 24),
                                  ).to(DEVICE)
 
    def forward(self, x_v, x_t):
        batch_size = x_v.size(0)
        x_v_sum = x_v.sum(dim=2, )[:,-24:]
        
        x_v = (1+x_v).log()
        x = torch.cat((x_v, x_t), dim=2)
        
        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r,0))

        
        #skip-rnn
        
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2,0,3,1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r,s),1)
        
        res = self.linear1(r)
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1,self.m)
            res = res + z
            
        if (self.output):
            res = self.output(res)
        
        res = self.lin(res)
        res *= x_v_sum
            
        return res

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