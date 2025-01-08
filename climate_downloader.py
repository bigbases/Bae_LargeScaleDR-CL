import time, re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver

class downloader:
        
    def __init__(self, driver_path, example_url):
        
        self.driver = webdriver.Chrome(driver_path)
        self.base_url = self._get_base_climate_url(example_url)
        self.wait_time = 0
        self.col_size = 10
    
    def _get_base_climate_url(self, url):
        
        base_climate_url = re.sub('^(https://www.wunderground.com/history/daily/.+/date)/.+$', '\\1', url)
        return base_climate_url
    
    
    def _date_parser(self, date):
        
        d_date = datetime.fromisoformat(date)
        ret = '%s-%s-%s' % (d_date.year, d_date.month, d_date.day)
        return ret
    
    def _get_valid_time(self, input_time):
        
        compare_seq = input_time.shift(1, fill_value = input_time.min() - timedelta(days = 2))
        d_grp = np.cumsum(np.where(input_time >= compare_seq, 0, 1))
        grp, cnt = np.unique(d_grp, return_counts=True)
        major_group = grp[np.argmax(cnt)]
        d_zero = input_time
        d_plus = input_time + timedelta(days=1)
        d_minus = input_time - timedelta(days=1)
        input_time = np.where(d_grp > major_group, d_plus, np.where(d_grp < major_group, d_minus, d_zero)) 
        return input_time    
            
    def get_data(self, date):
        
        url_date = self._date_parser(date)
        
        target_url = self.base_url + '/' + url_date
        self.driver.get(target_url)
        data_list = []
        
        self.wait_time = 0
        while(len(data_list)==0):
            time.sleep(0.5)
            self.wait_time += 0.5
            table_element = self.driver.find_element_by_tag_name('lib-city-history-observation')
            data_list = [item.text for item in table_element.find_elements_by_xpath(".//*[self::td or self::th]")]
        
        row_size = int(len(data_list)/self.col_size)
        data_arr = np.array(data_list).reshape(row_size, self.col_size)
        
        data_header = data_arr[0]
        data_body = data_arr[1:]
        data_df = pd.DataFrame(data_body, columns = data_header)            
        data_df['time'] = pd.to_datetime(data_df['Time'].apply(lambda x: date + ' ' + x))
        data_df['time'] = self._get_valid_time(data_df['time'])
        
        data_df['temperature']=data_df['Temperature'].str.replace(' *F *$', '').astype(float)
        data_df['dew_point']=data_df['Dew Point'].str.replace(' *F *$', '').astype(float)
        data_df['humidity']=data_df['Humidity'].str.replace(' *% *$', '').astype(float)
        data_df['wind']=data_df['Wind']
        data_df['wind_speed']=data_df['Wind Speed'].str.replace(' *mph *$', '').astype(float)
        data_df['wind_gust']=data_df['Wind Gust'].str.replace(' *mph *$', '').astype(float)
        data_df['pressure']=data_df['Pressure'].str.replace(' *in *$', '').astype(float)
        data_df['precip']=data_df['Precip.'].str.replace(' *in *$', '').astype(float)
        data_df['condition']=data_df['Condition']        

        ret_cols = ['time', 'temperature', 'dew_point', 'humidity', 'wind', 
                    'wind_speed', 'wind_gust', 'pressure', 'precip', 'condition']
        data_df = data_df[ret_cols]        
        return data_df
    
    def close(self):
        
        self.driver.close()