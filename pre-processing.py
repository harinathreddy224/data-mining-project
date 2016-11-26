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

print(dfSP500.describe())
plt.plot(dfSP500['Date'], dfSP500['Close'])
# pylab.show()

minOrMax = peakdetect(dfSP500['Close'].as_matrix(), lookahead=50)[0]

print(minOrMax)

labels = []

for index, peak in enumerate(minOrMax[:-1]):
	peakValue = peak[1]
	lastPeakValue = minOrMax[index + 1][1]
	delta = peakValue - lastPeakValue
	if delta > 0:
		labels.append("bull")
	else:
		if abs((delta / lastPeakValue) * 100 ) > 20:
			labels.append("bear")

print("Bull:", labels.count("bull"))

print("Bear:", labels.count("bear"))