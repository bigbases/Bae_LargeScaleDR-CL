import pandas as pd
import numpy as np
import pickle, pathlib
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
# from pyspark.sql.types import IntegerType, FloatType, StringType, ArrayType
# from pyspark.sql.functions import udf

# 시간 코드 해석
def parsing_cer_time_code(code):
    code_s = str(code)
    days, tick = int(code_s[:3])-1, int(code_s[3:])-1
    date = datetime(2009, 1, 1) + timedelta(days=days)
    hms = timedelta(minutes=30*tick)
    time = date + hms
    return time

# 시간 코드 생성
def generate_cer_time_code(start_time, end_time): # 시작, 끝 시간 입력
    t_seq = pd.date_range(start_time, end_time, freq='30T').to_series() # 30분 간격 Series 생성
    t_diff = t_seq - datetime(2009, 1, 1) # 2009.1.1 과의 간격 계산
    t_df = t_diff.dt.components # days,hours,minutes,seconds,milliseconds,microseconds,nanoseconds 구분한 df
    t_df.index.name = 'time'
    t_df['days_s']=(t_df['days'] + 1).astype(str) # days + 1 하는 이유는??
    t_df['tick_s']=(t_df['hours']+t_df['minutes']/30+1).astype(int).astype(str).str.zfill(2)
    t_df['code']=t_df['days_s']+t_df['tick_s'] # 시간 코드 생성
    t_df = t_df.reset_index()
    t_df = t_df.loc[:, ['time', 'code']]

    return t_df

spark = SparkSession.builder.getOrCreate()

raw_fold = pathlib.Path('D:/data/CER')
power_raw_fold = raw_fold/'power_raw'
allocation_file = raw_fold/'allocations.csv'

save_fold = pathlib.Path('data')/'values'
save_fold.mkdir(exist_ok=True)

POWER_RAW_SCHEMA = 'id string, code string, value float'

power_raw_files = [x.as_posix() for x in power_raw_fold.glob('*')]
raw_pwr_sdf = spark.read.csv(power_raw_files, sep=' ', schema=POWER_RAW_SCHEMA)
raw_pwr_sdf.createOrReplaceTempView('raw_pwr_view')

allo_sdf = spark.read.csv(allocation_file.as_posix(), header=True)
allo_sdf = allo_sdf[['id', 'code']]
allo_sdf.createOrReplaceTempView('allo_view')

spark.sql("""DROP TABLE IF EXISTS raw_pwr_tbl""")
spark.sql("""CACHE TABLE raw_pwr_tbl AS 
SELECT t1.id, t1.code, t1.value FROM
(SELECT id, code, value FROM raw_pwr_view) AS t1
JOIN (SELECT DISTINCT id FROM allo_view WHERE code IN ('1', '2')) AS t2
ON t1.id = t2.id""")

code_range = spark.sql("""SELECT MIN(code) AS start_code, MAX(code) AS end_code FROM raw_pwr_tbl""").toPandas()
start_code, end_code = code_range.loc[0, ['start_code', 'end_code']]

start_time = parsing_cer_time_code(start_code)
end_time = parsing_cer_time_code(end_code)
t_df = generate_cer_time_code(start_time, end_time)

ids = spark.sql("""SELECT DISTINCT id FROM raw_pwr_tbl ORDER BY id""").toPandas()
ids = ids['id'].to_list()

for target_id in ids:
# target_id = ids[0]

    save_file = save_fold/'H{:s}.bin'.format(target_id)
    if save_file.exists():
        continue

    id_df = spark.sql("""SELECT code, value FROM raw_pwr_tbl WHERE id == '{:s}'""".format(target_id))
    id_df = id_df.toPandas()
    id_df = id_df.drop_duplicates('code')
    id_df = pd.merge(t_df, id_df, on='code', how='left')
    id_df['id'] = target_id
    id_df = id_df.loc[:,['value']]
    id_df['value'] = id_df['value'].interpolate()
    id_df['value'] = id_df['value'].fillna(method='bfill')
    id_df = id_df.reset_index(drop=True)
    id_df = id_df.rename(columns={'value':'h{:s}'.format(target_id)})

    with open(save_file.as_posix(), 'wb') as f:
        pickle.dump(id_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Save as', save_file.as_posix())

time_df = t_df.drop('code', axis=1)

def create_time_feature(x):
    
    WEEKDAY_DICT = {0:'mon', 1:'tue', 2:'wed', 3:'thu', 4:'fri', 5:'sat', 6:'sun'}
    
    df = pd.DataFrame({'date':x.dt.date, 'weekday_':x.dt.weekday, 
                       'year_':x.dt.year, 'month_':x.dt.month, 'day_':x.dt.day,
                       'hour_':x.dt.hour, 'minute_':x.dt.minute}, index=x.index)

    s_d = datetime(df['year_'].min(), 1, 1)
    e_d = datetime(df['year_'].max(), 12, 31)
    d_s = pd.Series(pd.date_range(s_d, e_d, freq='1D'))
    d_df=d_s.groupby([d_s.dt.year, d_s.dt.month]).count().reset_index()
    d_df.columns = ['year_', 'month_', 'daysofmonth_']
    df = pd.merge(df, d_df, on = ['year_', 'month_'], how='inner')
    df = df.astype({'month_':float, 'day_':float, 'hour_':float, 'minute_':float, 'daysofmonth_':float})
    
    f = lambda x: (np.sin(2*np.pi*x), np.cos(2*np.pi*x))
    df['minute_x'], df['minute_y'] = f(df['minute_'] / 60.0)
    df['hour_x'], df['hour_y'] = f((df['hour_'] + df['minute_'] / 60.0) / 24.0)
    df['day_x'], df['day_y'] = f(df['day_']/df['daysofmonth_'])
    df['month_x'], df['month_y'] = f(df['month_']/12.0)

    w_df = pd.get_dummies(df['weekday_'].replace(WEEKDAY_DICT))[WEEKDAY_DICT.values()]
    w_df = w_df.rename(columns = lambda x: x+'_yn')
    w_df = w_df.drop(columns = ['sun_yn'])
    df = pd.concat([df, w_df], axis=1)
    df['time']=x
    df = df[['time', 'date', 'minute_x', 'minute_y', 'hour_x', 'hour_y', 'day_x', 'day_y', 'month_x', 'month_y', 
            'mon_yn', 'tue_yn', 'wed_yn', 'thu_yn', 'fri_yn', 'sat_yn']]
    
    return df

time_df = create_time_feature(time_df['time'])

climate_file = pathlib.Path('data')/'raw'/'climate_history'/'ireland.bin'

with open(climate_file.as_posix(), 'rb') as f:
    c_df = pickle.load(f)

time_df = pd.merge(time_df, c_df, on='date', how='left').sort_values('time').reset_index(drop=True)

# https://www.timeanddate.com/holidays/ireland
holidays = [datetime(2009, 1, 1), # New Year's Day
            datetime(2009, 3, 17), # St. Patrick's Day 
            datetime(2009, 4, 10), # Good Friday
            datetime(2009, 4, 12), # Easter
            datetime(2009, 4, 13), # Easter Monday
            datetime(2009, 5, 4), # May Day
            datetime(2009, 6, 1), # June Bank Holiday
            datetime(2009, 8, 3), # August Bank Holiday
            datetime(2009, 10, 26), # October Bank Holiday 
            datetime(2009, 12, 24), # Christmas Eve
            datetime(2009, 12, 25), # Christmas Day
            datetime(2009, 12, 26), # St. Stephen's Day
            datetime(2009, 12, 31), # New Year's Eve
            datetime(2010, 1, 1), # New Year's Day
            datetime(2010, 3, 17), # St. Patrick's Day
            datetime(2010, 4, 2), # Good Friday
            datetime(2010, 4, 4), # Easter
            datetime(2010, 4, 5), # Easter Monday
            datetime(2010, 5, 3), # May Day
            datetime(2010, 6, 7), # June Bakn Holiday
            datetime(2010, 8, 2), # August Bank Holiday
            datetime(2010, 10, 25), # October Nank Holiday
            datetime(2010, 12, 24), # Christmas Eve
            datetime(2010, 12, 25), # Christmas Day
            datetime(2010, 12, 26), # St. Stephen's Day
            datetime(2010, 12, 31), # New Year's Eve
            datetime(2011, 1, 1), # New Year's Day
            datetime(2011, 3, 17), # St. Patrick's Day
            datetime(2011, 4, 22), # Good Friday
            datetime(2011, 4, 24), # Easter
            datetime(2011, 4, 25), # Easter Monday
            datetime(2011, 5, 2), # May Day
            datetime(2011, 6, 6), # June Bank Holiday
            datetime(2011, 8, 1), # August Bank Holiday
            datetime(2011, 10, 31), # October Bank Holiday
            datetime(2011, 12, 24), # Christmas Eve
            datetime(2011, 12, 25), # Christmas Day
            datetime(2011, 12, 26), # St. Stephen's Day
            datetime(2011, 12, 31) # New Year's Eve
           ]
holidays = pd.Series(holidays)

time_df['holiday_yn'] = np.where(time_df['date'].isin(holidays.dt.date), 1, 0)

def read_bin(filepath):
    with open(filepath, 'rb') as f:
        df = pickle.load(f)
    return df

h_dfs = [read_bin(x.as_posix()) for x in save_fold.glob('*')]
h_df = pd.concat(h_dfs, axis=1)
a_df = pd.concat([time_df, h_df], axis=1)

fin_save_file = pathlib.Path('data')/'master.bin'
with open(fin_save_file.as_posix(), 'wb') as f:
    pickle.dump(a_df, f, protocol=pickle.HIGHEST_PROTOCOL)