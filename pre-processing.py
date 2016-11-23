print("Start pre-processing...")

import pandas as pd
import numpy as np

folderPath = "./data/"

# Process subset data

df = pd.read_csv('./dataset/sub-set.csv')

df['date'] = pd.to_datetime(df['date'])

df = df[(df['date'].dt.year == 2016)]

df = df[['ticker','date', 'close']]

tickers = df['ticker'].unique()


# Process S&P 500

dfSP500 = pd.read_csv('./dataset/SP500.csv')

dfSP500['Date'] = pd.to_datetime(dfSP500['Date'])

dfSP500 = dfSP500[(dfSP500['Date'].dt.year == 2016)]

dfSP500 = dfSP500[['Date', 'Close']]

print(dfSP500)


for ticker in tickers:
	stockDf = df.loc[df['ticker'] == ticker]
	last200days = stockDf.tail(200)
	first100days = last200days.head(100)
	last100days = last200days.tail(100)
	firstDayClose = last100days.head(1)['close']
	lastDayClose = last100days.tail(1)['close']
	delta = np.float32(lastDayClose) - np.float32(firstDayClose)
	percentageChange = (delta / np.float32(lastDayClose)) * 100
	if delta[0] > 0:
		filename = folderPath + ticker + "-up" + ".csv"
	else:
		filename = folderPath + ticker + "-down" +".csv"
	text_file = open(filename, "w")
	df.to_csv(filename)
	text_file.close()
