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

# HSI
HSI = Share('^HSI') 
filePathHSI = './data/hsi.csv'
# historyHSI = HSI.get_historical(startDay, endDay)
# dfHSI = pd.DataFrame(historyHSI)
dfHSI = pd.read_csv(filePathHSI)
dfHSI['Date'] = pd.to_datetime(dfHSI['Date'])
dfHSI = dfHSI.sort_values(by='Date', ascending=False)
# dfHSI.to_csv(filePathHSI)

# print(dfHSI.describe())

# SP500
SP500 = Share('^GSPC')
filePathSP500 = './data/sp500.csv'
# historySP500 = SP500.get_historical(startDay, endDay)
# dfSP500 = pd.DataFrame(historySP500)
dfSP500 = pd.read_csv(filePathSP500)
dfSP500['Date'] = pd.to_datetime(dfSP500['Date'])
dfSP500 = dfSP500.sort_values(by='Date', ascending=False)
# dfSP500.to_csv(filePathSP500)

# print(dfSP500.describe())

# Gold
filePathGold = './data/gold-price.csv'
dfGold = pd.read_csv(filePathGold)
dfGold['Date'] = pd.to_datetime(dfGold['Date'])
dfGold = dfGold.sort_values(by='Date', ascending=False)
mask = (dfGold['Date'] > startDay) & (dfGold['Date'] <= endDay)
dfGold = dfGold.loc[mask]

print(dfGold.describe())

# # Oil price
filePathOil = './data/oil-price.csv'
dfOil = pd.read_csv(filePathOil)
dfOil['DATE'] = pd.to_datetime(dfOil['DATE'])
dfOil = dfOil.sort_values(by='DATE', ascending=False)
dfOil = dfOil[dfOil['VALUE'] != "."]

# print(dfOil.describe())

dfHSI["label"] = np.where(dfHSI['Close'] > dfHSI['Open'], '+1', '-1')

dfHSI['Close'] = dfHSI['Close'].apply(lambda x: np.log(x))  

dfHSI['feature1'] = dfHSI['Close'] > dfHSI['Close'].shift()
dfHSI['feature2'] = dfHSI['Close'] > dfHSI['Close'].shift(2)

dfHSI['feature3'] = dfSP500['Close'] > dfSP500['Close'].shift()
dfHSI['feature4'] = dfSP500['Close'] > dfSP500['Close'].shift(2)

dfHSI['feature5'] = dfGold['Value'] > dfGold['Value'].shift()
dfHSI['feature6'] = dfGold['Value'] > dfGold['Value'].shift(2)

dfHSI['feature7'] = dfOil['VALUE'] > dfOil['VALUE'].shift()
dfHSI['feature8'] = dfOil['VALUE'] > dfOil['VALUE'].shift(2)

# print(dfHSI)

dfHSI = dfHSI.ix[2:]
dfHSI = dfHSI.fillna(0)

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
