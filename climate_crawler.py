import os
import pandas as pd
import pickle as p
from datetime import datetime, timedelta

import climate_downloader as cd

driver_path = 'C:/ChromeDriver/chromedriver'
example_url = 'https://www.wunderground.com/history/daily/ie/clenagh/EINN/date/2020-3-14'
dwr = cd.downloader(driver_path, example_url)

start_date = '2009-01-01' 
end_date = '2010-06-01' 

start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

span_delta = end_date_dt - start_date_dt 
download_date_list = [] 
for days in range(span_delta.days + 1): 
    target_date_dt = start_date_dt + timedelta(days = days) 
    target_date_str = target_date_dt.date().isoformat() 
    download_date_list.append(target_date_str) 

save_fold = os.path.dirname(os.path.abspath(__file__))
for download_date in download_date_list:
    save_file = download_date + '.bin' 
    save_path = save_fold / save_file 
    
    if os.path.exists(save_path): 
        continue 

        
    data_df = dwr.get_data(download_date) 

    with open(save_path.as_posix(), 'wb') as f:
        p.dump(data_df, f, protocol=p.HIGHEST_PROTOCOL)
    
    print(download_date)

data_df = dwr.get_data(date)

import os
import pickle
save_fold = 'C:/tmp/climate_history/ireland'
os.makedirs(save_fold, exist_ok=True)

dcd = daily_climate_downloader('ie/clenagh/EINN')

a, b = dcd.get_data('2009-12-08')

for download_date in download_date_list:
    save_file = download_date + '.bin' 
    save_path = save_fold + '/' + save_file 
    
    if os.path.exists(save_path): 
        continue 
        
    data = dcd.get_data(download_date) 
   
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

save_file_list = os.listdir(save_fold)
save_file_list

dcd.close()

data_df_list = [] 
for save_file in save_file_list:
    load_path = save_fold + '/' + save_file
    with open(load_path, 'rb') as f:
        tmp_df = pickle.load(f) 
        data_df_list.append(tmp_df) 

data_df = pd.concat(data_df_list)
data_df = data_df.reset_index(drop = True)