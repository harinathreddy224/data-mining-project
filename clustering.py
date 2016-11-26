print("Start clustering...")

from yahoo_finance import Share
import pandas as pd

url = "%s%s" % (_HISTORICAL_GOOGLE_URL,
                urlencode({"q": sym,
                           "startdate": start.strftime('%b %d, ' '%Y'),
                           "enddate": end.strftime('%b %d, %Y'),
                           "output": "csv"}))

# startDay = '2001-01-01'
# endDay = '2010-12-31'
# stock = Share('0960.HK') 
# filePath = './data/ckh.csv'

# # # Get data and save to csv
# history = stock.get_historical(startDay, endDay)
# df = pd.DataFrame(history)
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.sort_values(by='Date', ascending=False)
# df.to_csv(filePath)


