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
endDay = '2015-12-31'
today = datetime.strftime(datetime.today(), "%Y-%m-%d")
stock = Share('^HSI') 
filePath = './data/hsi.csv'

# # Get data and save to csv
history = stock.get_historical(startDay, endDay)
df = pd.DataFrame(history)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date', ascending=False)
df.to_csv(filePath)

# Plot graph
# plt.plot(df['Close'])
# pylab.show()

# Add label
df = pd.read_csv(filePath)
df["label"] = np.where(df['Close'] > df['Open'], '+1', '-1')

df['Close'] = df['Close'].apply(lambda x: np.log(x))  

df['feature1'] = df['Close'] > df['Close'].shift()
df['feature2'] = df['Close'] > df['Close'].shift(2)
df['feature3'] = df['Close'] > df['Close'].shift(3)
df['feature4'] = df['Close'] > df['Close'].shift(4)
df['feature5'] = df['Close'] > df['Close'].shift(5)
df['feature6'] = df['Close'] > df['Close'].shift(6)
df['feature7'] = df['Close'] > df['Close'].shift(7)
df['feature8'] = df['Close'] > df['Close'].shift(8)
df['feature9'] = df['Close'] > df['Close'].shift(9)
df['feature10'] = df['Close'] > df['Close'].shift(10)
df['feature11'] = df['Close'] > df['Close'].shift(11)
df['feature12'] = df['Close'] > df['Close'].shift(12)
df['feature13'] = df['Close'] > df['Close'].shift(13)
df['feature14'] = df['Close'] > df['Close'].shift(14)
df['feature15'] = df['Close'] > df['Close'].shift(15)
df['feature16'] = df['Close'] > df['Close'].shift(16)

df = df.ix[16:]

numberOfRow = df.shape[0]

print(numberOfRow)
numberOfTrain = int(round(numberOfRow * 0.8))
numberOfTest = int(round(numberOfRow * 0.2))

print(numberOfTrain)

train = df.head(n=numberOfTrain)
test = df.tail(n=numberOfTest)

# MLP trains on two arrays: 
# array X of size (n_samples, n_features), which holds the training samples represented as floating point feature vectors
# array y of size (n_samples,), which holds the target values (class labels) for the training samples
X = train[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16"]]
y = train["label"]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)     

X_test = test[["feature1", "feature2", "feature3", "feature4", "feature5", "feature6", "feature7", "feature8", "feature9", "feature10", "feature11", "feature12", "feature13", "feature14", "feature15", "feature16"]]
y_true = test["label"]

# After fitting (training), the model can predict labels for new samples
y_pred = clf.predict(X_test)

print(accuracy_score(y_true, y_pred))
