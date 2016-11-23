print("Start pre-processing...")

import pandas as pd

folderPath = "./data/"

df = pd.read_csv('./dataset/sub-set.csv')

df['date'] = pd.to_datetime(df['date'])

df = df[(df['date'].dt.year == 2016)]

df = df[['ticker','date', 'close']]

tickers = df['ticker'].unique()

print(tickers)

for ticker in tickers:
	filename = ticker + ".txt"
	text_file = open(folderPath + filename, "w")
	text_file.write("something")
	text_file.close()
