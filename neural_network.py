print("Start neural network...")

from yahoo_finance import Share
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
# Scikit-learn is a machine learning library 
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

startDay = '2001-01-01'
endDay = '2010-12-31'

# # HSI
HSI = Share('^HSI') 
filePathHSI = './data/hsi.csv'
historyHSI = HSI.get_historical(startDay, endDay)
dfHSI = pd.DataFrame(historyHSI)
# dfHSI = pd.read_csv(filePathHSI)
dfHSI['Date'] = pd.to_datetime(dfHSI['Date'])
dfHSI = dfHSI.sort_values(by='Date', ascending=False)
dfHSI.to_csv(filePathHSI)

# # SP500
SP500 = Share('^GSPC')
filePathSP500 = './data/sp500.csv'
historySP500 = SP500.get_historical(startDay, endDay)
dfSP500 = pd.DataFrame(historySP500)
# dfSP500 = pd.read_csv(filePathSP500)
dfSP500['Date'] = pd.to_datetime(dfSP500['Date'])
dfSP500 = dfSP500.sort_values(by='Date', ascending=False)
dfSP500 = dfSP500.rename(columns = {'Open':'OpenSP'})
dfSP500 = dfSP500.rename(columns = {'Close':'CloseSP'})
dfSP500.to_csv(filePathSP500)

# # Gold
filePathGold = './data/gold-price.csv'
# dfGold = pd.read_csv(filePathGold)
# dfGold['Date'] = pd.to_datetime(dfGold['Date'])
# dfGold = dfGold.sort_values(by='Date', ascending=False)
# mask = (dfGold['Date'] > startDay) & (dfGold['Date'] <= endDay)
# dfGold = dfGold.loc[mask]


# # # Oil price
filePathOil = './data/oil-price.csv'
# dfOil = pd.read_csv(filePathOil)
# dfOil['DATE'] = pd.to_datetime(dfOil['DATE'])
# dfOil = dfOil.sort_values(by='DATE', ascending=False)
# dfOil = dfOil[dfOil['VALUE'] != "."]

# # Merge by date
# dfHSI.index = dfHSI['Date']
# dfSP500.index = dfSP500['Date']
# dfGold.index = dfGold['Date']
# dfOil.index = dfOil['DATE']

# # print("dfHSI.describe()")
# # print(dfHSI.describe())

# # print("dfSP500.describe()")
# # print(dfSP500.describe())

# # print("dfGold.describe()")
# # print(dfGold.describe())

# # print("dfOil.describe()")
# # print(dfOil.describe())

# dfHSI = dfHSI.merge(dfSP500, left_on = 'Date', right_on = 'Date', how = 'inner')
# dfHSI = dfHSI.merge(dfGold, left_on = 'Date', right_on = "Date", how = "inner")
# dfHSI = dfHSI.merge(dfOil, left_on = 'Date', right_on = "DATE", how = "inner")

# print(dfHSI)

sp500 = pd.read_csv(filePathSP500)
gold = pd.read_csv(filePathGold)
oil = pd.read_csv(filePathOil)
hsi = pd.read_csv(filePathHSI)

sp500['Date'] = pd.to_datetime(sp500['Date'])
gold['Date'] = pd.to_datetime(gold['Date'])
oil['DATE'] = pd.to_datetime(oil['DATE'])
hsi['Date'] = pd.to_datetime(hsi['Date'])

gold.index = gold['Date']
oil.index = oil['DATE']
hsi.index = hsi['Date']
sp500.index = sp500['Date']

hsi = hsi.merge(sp500, left_on = 'Date', right_on = 'Date', how = 'inner')
hsi = hsi.merge(gold, left_on = 'Date', right_on = "Date", how = "inner")
hsi = hsi.merge(oil, left_on = 'Date', right_on = "DATE", how = "inner")

hsi.to_csv("./data/merged.csv")

dfHSI = hsi

dfHSI["label"] = np.where(dfHSI['Close'] > dfHSI['Open'], '+1', '-1')

dfHSI['Close'] = dfHSI['Close'].apply(lambda x: np.log(x))  
dfHSI['CloseSP'] = dfHSI['CloseSP'].apply(lambda x: np.log(x)) 

dfHSI['feature1'] = dfHSI['Close'] > dfHSI['Close'].shift()
dfHSI['feature2'] = dfHSI['Close'] > dfHSI['Close'].shift(2)

dfHSI['feature3'] = dfHSI['CloseSP'] > dfHSI['CloseSP'].shift()
dfHSI['feature4'] = dfHSI['CloseSP'] > dfHSI['CloseSP'].shift(2)

dfHSI['feature5'] = dfHSI['GOLD'] > dfHSI['GOLD'].shift()
dfHSI['feature6'] = dfHSI['GOLD'] > dfHSI['GOLD'].shift(2)

dfHSI['feature7'] = dfHSI['OIL'] > dfHSI['OIL'].shift()
dfHSI['feature8'] = dfHSI['OIL'] > dfHSI['OIL'].shift(2)

# print(dfHSI)

dfHSI = dfHSI.ix[2:]
# dfHSI = dfHSI.fillna(0)

numberOfRow = dfHSI.shape[0]
numberOfTrain = int(round(numberOfRow * 0.8))
numberOfTest = int(round(numberOfRow * 0.2))

train = dfHSI.head(n=numberOfTrain)
test = dfHSI.tail(n=numberOfTest)

# MLP trains on two arrays: 
# array X of size (n_samples, n_features), which holds the training samples represented as floating point feature vectors
# array y of size (n_samples,), which holds the target values (class labels) for the training samples
# X = train[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16"]]
X = train[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8"]]
y = train["label"]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)     

# X_test = test[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16"]]
X_test = test[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8"]]
y_true = test["label"]

# After fitting (training), the model can predict labels for new samples
y_pred = clf.predict(X_test)

print(accuracy_score(y_true, y_pred))
