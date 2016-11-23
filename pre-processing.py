print("Start pre-processing...")

import pandas as pd
import numpy as np

folderPath = "./data/"

df = pd.read_csv('./dataset/sub-set.csv')

df['date'] = pd.to_datetime(df['date'])

df = df[(df['date'].dt.year == 2016)]

df = df[['ticker','date', 'close']]

tickers = df['ticker'].unique()

print(tickers)

for ticker in tickers:
	filename = ticker + ".csv"
	stockDf = df.loc[df['ticker'] == ticker]
	last200days = stockDf.tail(200)
	first100days = last200days.head(100)
	last100days = last200days.tail(100)
	firstDayClose = last100days.head(1)['close']
	lastDayClose = last100days.tail(1)['close']
	delta = np.float32(lastDayClose) - np.float32(firstDayClose)
	if delta[0] > 0:
		print("positive")
	else:
		print("negative")
	print(delta)
	text_file = open(folderPath + filename, "w")
	df.to_csv(folderPath + filename)
	text_file.close()
