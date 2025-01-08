import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


class IndexSetup:
    
    def __init__(self, conf, data):
        self.conf = conf
        self.prev_times, self.next_times = conf['prev_times'], conf['next_times']
        self.season_index = self._get_season_index(data)
        self.base_index_set = self._get_base_index_set(self.season_index) # 뭘 의미하는 걸까?
        self.test_start_index = self._get_test_start_index(data, self.base_index_set)
        self.train_period_index = self._get_train_period_index(self.season_index, self.test_start_index)
        self.test_period_index = self._get_test_period_index(self.season_index, self.test_start_index)
        self.base_index_set_train = self._get_base_index_set_train(self.base_index_set, self.test_start_index)
        self.base_index_set_test = self._get_base_index_set_test(self.base_index_set, self.test_start_index)

       
    def _get_season_index(self, data):
        conf = self.conf
        time_tag, summer_months, season = conf['time_flag_var'], conf['summer_months'], conf['season']
        
        summer_mask = data[time_tag].dt.month.isin(summer_months)
        season_index = data.index[summer_mask] if season == 'summer' else data.index[~summer_mask]
        self._check_season_index_vailidity(season_index)
        
        return season_index

    
    def _check_season_index_vailidity(self, season_index):
        prev_times, next_times = self.prev_times, self.next_times

        index_diff = pd.Series(season_index).diff().drop_duplicates().dropna()
        maximum_time_size = index_diff[index_diff > 1].min()
        
        assert prev_times < maximum_time_size, """too large 'prev_step'."""
        assert next_times < maximum_time_size, """too large 'next_step'."""

                    
    def _get_base_index_set(self, season_index):
        prev_times, next_times = self.prev_times, self.next_times
        
        season_index1 = season_index[season_index.isin(season_index + prev_times)]
        season_index2 = season_index[season_index.isin(season_index - next_times)]
        base_index_set = season_index1[season_index1.isin(season_index2)]
        
        return base_index_set
    
    
    def _get_test_start_index(self, data, base_index_set):
        conf = self.conf
        time_tag, test_ratio = conf['time_flag_var'], conf['test_ratio']
        
        qtfit = QuantileTransformer(n_quantiles = len(base_index_set))
        
        idxdf = data.loc[base_index_set, [time_tag]]
        idxdf['hour'], idxdf['minute'] = idxdf[time_tag].dt.hour, idxdf[time_tag].dt.minute
        idxdf['q'] = qtfit.fit_transform(idxdf.index.values.reshape(-1, 1)).flatten()
        
        mask = (idxdf['q'] > 1 - test_ratio) & (idxdf['hour'] == 0) & (idxdf['minute'] == 0)
        test_start_index = idxdf.loc[mask].index.min()
        
        return test_start_index
    
    
    def _get_train_period_index(self, season_index, test_start_index):
        train_period_index = season_index[season_index < test_start_index]
        
        return train_period_index
    

    def _get_test_period_index(self, season_index, test_start_index):
        test_period_index = season_index[season_index >= test_start_index]
        
        return test_period_index
    
    
    def _get_base_index_set_train(self, base_index_set, test_start_index):
        conf = self.conf
        train_sample_ratio = conf['train_sample_ratio']
        
        train_sample_size = int(len(base_index_set) * train_sample_ratio)
        base_index_set_train = base_index_set[base_index_set < test_start_index]
        
        if train_sample_ratio < 1:
            base_index_set_train = np.random.choice(base_index_set_train, train_sample_size)
            base_index_set_train = pd.Int64Index(base_index_set_train).sort_values()
        
        return base_index_set_train
    
    
    def _get_base_index_set_test(self, base_index_set, test_start_index):
        next_times = self.next_times
        
        base_index_set_test_pre = base_index_set[base_index_set >= test_start_index-1] 
        interval = np.arange(len(base_index_set_test_pre)) % next_times == 0
        base_index_set_test = base_index_set_test_pre[interval]
    
        return base_index_set_test
    
    
    def get_indexes(self, base_index):
        prev_times, next_times = self.prev_times, self.next_times
        
        # prev_index = pd.Int64Index(np.arange(base_index-prev_times, base_index) + 1)
        # next_index = pd.Int64Index(np.arange(base_index, base_index+next_times) + 1)   
        prev_index = pd.Index(np.arange(base_index-prev_times, base_index) + 1, dtype="int64") 
        next_index = pd.Index(np.arange(base_index, base_index+next_times) + 1, dtype="int64") 
        
        return prev_index, next_index
        
        
    
class ColumnsSetup:
    
    def __init__(self, conf, data):
        self.conf = conf
        self.use_var_names, self.time_var_names, self.value_var_names = self._get_use_var_names(data)
        self.use_var_locs, self.time_var_locs, self.value_var_locs = self._get_use_var_locs(data)
        self.base_var_locs, self.base_var_names = self._get_base_var_infos(data)
            
            
    def _get_use_var_names(self, data):
        conf = self.conf
        all_var_names = data.columns
        time_var_format, value_var_format = conf['time_var_format'], conf['value_var_format']
        
        time_var_names = all_var_names[all_var_names.str.match(time_var_format)]
        value_var_names = all_var_names[all_var_names.str.match(value_var_format)]
        use_var_names = time_var_names.append(value_var_names).drop_duplicates()
        
        return use_var_names, time_var_names, value_var_names

    
    def _get_use_var_locs(self, data):
        all_var_names = data.columns
        use_var_names = self.use_var_names 
        time_var_names, value_var_names = self.time_var_names, self.value_var_names
        
        time_var_locs = pd.Index(np.where(use_var_names.isin(time_var_names))[0])
        value_var_locs = pd.Index(np.where(use_var_names.isin(value_var_names))[0])
        use_var_locs = pd.Index(np.where(all_var_names.isin(use_var_names))[0])
        
        return use_var_locs, time_var_locs, value_var_locs
        

    def _get_base_var_infos(self, data):
        conf = self.conf
        time_flag_var = conf['time_flag_var']
        all_var_names = data.columns
        
        base_var_names = pd.Index([time_flag_var])
        base_var_locs = pd.Index(np.where(all_var_names.isin(base_var_names))[0])
        
        return base_var_locs, base_var_names