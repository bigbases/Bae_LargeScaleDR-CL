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
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

from timefeatures import time_features

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, conf, data, mode='train'):
        
        timeenc = 0 # 시간 인코딩 방식
        mark = pd.DataFrame()
        mark['date'] = data.time
        if timeenc == 0:
            mark['month'] = mark.date.apply(lambda row: row.month, 1)
            mark['day'] = mark.date.apply(lambda row: row.day, 1)
            mark['weekday'] = mark.date.apply(lambda row: row.weekday(), 1)
            mark['hour'] = mark.date.apply(lambda row: row.hour, 1)
            self.data_stamp = mark.drop(['date'], 1).values
        elif timeenc == 1:
            data_stamp = time_features(pd.to_datetime(mark['date'].values), freq=self.freq)
            self.data_stamp = data_stamp.transpose(1, 0)
                
        self.conf = conf
        self.index_setup = IndexSetup(conf, data)
        self.columns_setup = ColumnsSetup(conf, data)
        
        if mode == 'train':
            self.base_index_set = self.index_setup.base_index_set_train
            self
        else:
            self.base_index_set = self.index_setup.base_index_set_test
        
        self.use_var_size = len(self.columns_setup.use_var_names)
        self.time_var_size = len(self.columns_setup.time_var_names)
        self.value_var_size = len(self.columns_setup.value_var_names)
        self.prev_times = self.index_setup.prev_times
        self.next_times = self.index_setup.next_times
        
        self.values = torch.FloatTensor(pd.concat((data[["temp_max_d1_t_", "temp_min_d1_t_", "holiday_yn_d1_t_"]], data[self.columns_setup.value_var_names]), axis=1).values).to(DEVICE)
        self.times = torch.FloatTensor(data[self.columns_setup.time_var_names].values).to(DEVICE)
        self.indexs = torch.IntTensor(data.index).to(DEVICE)
                            
    def __len__(self):
        
        return len(self.base_index_set)
        
    def __getitem__(self, key):
        
        base_index = self.base_index_set[key]
        prev_index, next_index = self.index_setup.get_indexes(base_index)
        
        x = self.values[prev_index, :]
        # y = self.values[list(prev_index[-12:]) + list(next_index), :]
        y = self.values[list(prev_index[-12:]) + list(next_index), :]
        pos = self.indexs[list(prev_index[-12:]) + list(next_index)]
        
        x_mark = self.data_stamp[prev_index, :]
        y_mark = self.data_stamp[list(prev_index[-12:]) + list(next_index), :]
        
        return x, y, pos, x_mark, y_mark

class PowerModel(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, ds):
        super(PowerModel, self).__init__()
        self.seq_len = 192
        self.label_len = 24//2
        self.pred_len = 24
        self.output_attention = False

        # Decomp
        kernel_size = 11
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        x, y, pos, x_mark, y_mark = ds[0]
        self.a_size = x.shape[1]
        
        enc_in = self.a_size
        dec_in = self.a_size
        d_model = 512
        embed = "timeF"
        freq = "h"
        dropout = 0.05
        factor = 1
        output_attention = False # 어텐션 저장할 것인지
        n_heads = 8
        d_ff = 2048
        moving_avg = 11
        activation = "gelu"
        e_layers = 2
        c_out = self.a_size
        d_layers = 1
        
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq,
                                                  dropout).to(DEVICE)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq,
                                                  dropout).to(DEVICE)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        ).to(DEVICE)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        ).to(DEVICE)
        
        self.linear = nn.Sequential(nn.Linear(c_out, 256),
                                  nn.LeakyReLU(),
                                  nn.Linear(256, 1),
                                  ).to(DEVICE)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        dec_out = self.linear(dec_out)

        if self.output_attention:
            return dec_out[:, -self.pred_len:], attns
        else:
            return dec_out[:, -self.pred_len:]  # [B, L, D]

def training(conf, data, epoch_size=30, batch_size=16, lr=0.0001, shuffle=True):
    
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
                
        for x, y, pos, x_mark, y_mark in dl_train:
            
            x = (1+x).log()
            x_mark = x_mark.float().to(DEVICE)
            y_mark = y_mark.float().to(DEVICE)
            
            batch_count += 1
            model.train()
            model.zero_grad()
            
            # decoder input
            dec_inp = torch.zeros_like(y[:, -24:, :]).float()
            dec_inp = torch.cat([y[:, :12, :], dec_inp], dim=1).float().to(DEVICE)
        
            output = model(x, x_mark, dec_inp, y_mark)
                        
            loss = loss_function(output.flatten(), y[:,-24:,3:].sum(dim=2).flatten())
            err_train = loss.tolist()
            loss.backward()
            optimizer.step()

            model.eval()

            if not batch_count % 10:
                
                with torch.no_grad():

                    rets = []
                    for x, y, pos, x_mark, y_mark in dl_test:
                        
                        x = (1+x).log()
                        
                        x_mark = x_mark.float().to(DEVICE)
                        y_mark = y_mark.float().to(DEVICE)

                        # decoder input
                        dec_inp = torch.zeros_like(y[:, -24:, :]).float()
                        dec_inp = torch.cat([y[:, :12, :], dec_inp], dim=1).float().to(DEVICE)
                    
                        output = model(x, x_mark, dec_inp, y_mark)

                        ret = pd.DataFrame({'epoch':epoch_count, 
                                            'batch':batch_count, 
                                            'pos': pos[:,-24:].flatten().tolist(), 
                                            'act_val': y[:,-24:,3:].sum(dim=2).flatten().tolist(), 
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