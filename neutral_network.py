print("Start trading...")

from yahoo_finance import Share
import pandas as pd
import pylab
import matplotlib.pyplot as plt

startDay = '2001-01-01'
endDay = '2010-12-31'
stock = Share('^HSI') 

history = stock.get_historical(startDay, endDay)
df = pd.DataFrame(history)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date', ascending=True)
df.to_csv('./data/hsi.csv')

plt.plot(df['Close'])
pylab.show()