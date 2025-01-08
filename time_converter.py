import os, pathlib
import pandas as pd

datafd = pathlib.Path(os.environ['HOMEPATH']) / '__PYDATA__'
datafd.mkdir(parents=True, exist_ok=True)

outfd = pathlib.Path(os.environ['HOMEPATH']) / '__PYOUT__'
outfd.mkdir(parents=True, exist_ok=True)

basefd = datafd / 'base'
basefd.mkdir(parents=True, exist_ok=True)

class VariableController:
    def __init__(self, time_varnames_format='.+_t_$', value_varnames_format='^h[0-9]+'):
        self._tnf = time_varnames_format
        self._vnf = value_varnames_format
        
    def time_cols(self, data):
        return data.columns[data.columns.str.match(self._tnf)].tolist()
    
    def value_cols(self, data):
        return data.columns[data.columns.str.match(self._vnf)].tolist()

def transform_time_freq(df, freq):
    
    vcntr = VariableController()
    
    tmpdf = df.copy()
    tmpdf['time_new'] = tmpdf['time'].dt.floor(freq='1h')
    tmpdf = tmpdf.sort_values(['time']).reset_index(drop=True)

    tmpdf_v = tmpdf.groupby('time_new')[vcntr.value_cols(tmpdf)].sum().reset_index()
    tmpdf_t = tmpdf.groupby('time_new')[['time_new'] + vcntr.time_cols(tmpdf)].head(1)
    tmpdf_m = pd.merge(tmpdf_t, tmpdf_v).rename(columns={'time_new':'time'})
    tmpdf_m['date'] = tmpdf_m['time'].dt.date
    tmpdf_m = tmpdf_m[['time', 'date'] + vcntr.time_cols(tmpdf_m) + vcntr.value_cols(tmpdf_m)]
    tmpdf_m = tmpdf_m.sort_values('time').reset_index(drop=True)

    return tmpdf_m

pwrdf = pd.read_pickle(basefd / 'ireland_power_industry_master_20200712.bin')

pwrdf = transform_time_freq(pwrdf, freq='1h')

pwrdf.to_pickle(basefd / 'ireland_power_industry_1hour_20200808.bin')