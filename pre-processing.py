print("Start pre-processing...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import numpy as np
from peakdetect import peakdetect

folderPath = "./data/"

# Process S&P 500
dfSP500 = pd.read_csv('./dataset/SP500.csv')

dfSP500['Date'] = pd.to_datetime(dfSP500['Date'])

dfSP500 = dfSP500[['Date', 'Close']]

# print(dfSP500)
# plt.plot(dfSP500['Date'], dfSP500['Close'])
# pylab.show()

peaks = peakdetect(dfSP500['Close'].as_matrix(), lookahead=100)[0]

for index, peak in enumerate(peaks[:-1]):
	peakValue = peak[1]
	lastPeakValue = peaks[index + 1][1]
	delta = peakValue - lastPeakValue
	if delta > 0:
		print("up")
	else:
		percentageDrop = (delta / lastPeakValue) * 100
		if abs(percentageDrop) > 20.0:
			print("BEAR")


# 17.450001
# 16.98

# for ticker in tickers:
# 	stockDf = df.loc[df['ticker'] == ticker]
# 	last200days = stockDf.tail(200)
# 	first100days = last200days.head(100)
# 	last100days = last200days.tail(100)
# 	firstDayClose = last100days.head(1)['close']
# 	lastDayClose = last100days.tail(1)['close']
# 	delta = np.float32(lastDayClose) - np.float32(firstDayClose)
# 	percentageChange = (delta / np.float32(lastDayClose)) * 100
# 	if delta[0] > 0:
# 		filename = folderPath + ticker + "-up" + ".csv"
# 	else:
# 		filename = folderPath + ticker + "-down" +".csv"
# 	text_file = open(filename, "w")
# 	df.to_csv(filename)
# 	text_file.close()