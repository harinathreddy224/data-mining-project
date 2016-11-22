print("Start pre-processing...")

import pandas as pd

df = pd.read_csv('./dataset/sub-set.csv')


df['date'] = pd.to_datetime(df['date'])

df = df[(df['date'].dt.year == 2016) | (df['date'].dt.year == 2015)]

df = df[['ticker','date', 'close']]

print(df)