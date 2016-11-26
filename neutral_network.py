print("Start trading...")

from yahoo_finance import Share
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import numpy as np

# startDay = '2001-01-01'
# endDay = '2010-12-31'
# stock = Share('^HSI') 
filePath = './data/hsi.csv'

# Get data and save to csv
# history = stock.get_historical(startDay, endDay)
# df = pd.DataFrame(history)
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values(by='Date', ascending=True)
# df.to_csv(filePath)

# Plot graph
# plt.plot(df['Close'])
# pylab.show()

# Add outcome
df = pd.read_csv(filePath)
df["Outcome"] = np.where(df['Close'] > df['Open'], '+1', '-1')

print(df)