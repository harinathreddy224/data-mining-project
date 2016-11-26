print("Start trading...")

import graphlab as gl
from datetime import datetime
from yahoo_finance import Share
import pylab

# download historical prices of S&P 500 index
today = datetime.strftime(datetime.today(), "%Y-%m-%d")
stock = Share('^GSPC') # ^GSPC is the Yahoo finance symbol to refer S&P 500 index
# we gather historical quotes from 2001-01-01 up to today
hist_quotes = stock.get_historical('2001-01-01', today)

l_date = []
l_open = []
l_high = []
l_low = []
l_close = []
l_volume = []
# reverse the list
hist_quotes.reverse()
for quotes in hist_quotes:
    l_date.append(quotes['Date'])
    l_open.append(float(quotes['Open']))
    l_high.append(float(quotes['High']))
    l_low.append(float(quotes['Low']))
    l_close.append(float(quotes['Close']))
    l_volume.append(int(quotes['Volume']))

qq = gl.SFrame({'datetime' : l_date, 
          'open' : l_open, 
          'high' : l_high, 
          'low' : l_low, 
          'close' : l_close, 
          'volume' : l_volume})
# datetime is a string, so convert into datetime object
qq['datetime'] = qq['datetime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

qq.save("SP500_daily.bin")
# once data is saved, we can use the following instruction to retrieve it 
qq = gl.SFrame("SP500_daily.bin/")

import matplotlib.pyplot as plt
plt.plot(qq['close'])

pylab.show()
