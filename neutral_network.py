print("Start trading...")

from yahoo_finance import Share
import pandas as pd
import pylab

startDay = '2001-01-01'
endDay = '2010-12-31'
stock = Share('^HSI') 

hist_quotes = stock.get_historical(startDay, endDay)

df = pd.DataFrame(hist_quotes)

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by='Date', ascending=True)

print(df)

df.to_csv('hsi.csv')



# l_date = []
# l_open = []
# l_high = []
# l_low = []
# l_close = []
# l_volume = []
# # reverse the list
# hist_quotes.reverse()
# for quotes in hist_quotes:
#     l_date.append(quotes['Date'])
#     l_open.append(float(quotes['Open']))
#     l_high.append(float(quotes['High']))
#     l_low.append(float(quotes['Low']))
#     l_close.append(float(quotes['Close']))
#     l_volume.append(int(quotes['Volume']))

# hsi = gl.SFrame({'datetime' : l_date, 
#           'open' : l_open, 
#           'high' : l_high, 
#           'low' : l_low, 
#           'close' : l_close, 
#           'volume' : l_volume})

# df = pd.DataFrame(hsi)

# print(df)

# # datetime is a string, so convert into datetime object
# qq['datetime'] = qq['datetime'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d'))

# qq.save("SP500_daily.bin")
# # once data is saved, we can use the following instruction to retrieve it 
# qq = gl.SFrame("SP500_daily.bin/")

# import matplotlib.pyplot as plt
# plt.plot(qq['close'])

# pylab.show()
