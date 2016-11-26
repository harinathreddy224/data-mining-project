print("Start neural network...")

from yahoo_finance import Share
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import numpy as np
# Scikit-learn is a machine learning library 
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
# df['ho'] = df['High'] - df['Open'] # distance between Highest and Opening price
# df['lo'] = df['Low'] - df['Open'] # distance between Lowest and Opening price
# df['gain'] = df['Close'] - df['Open']

# df['feature1'] = df['Close'] > df['Close'].shift()
# df['feature2'] = df['Close'] > df['Close'].shift().shift()

df['feature1'] = df['Open']
df['feature2'] = df['Close']
df['feature3'] = df['Open'].shift().fillna('0')
df['feature4'] = df['Close'].shift().fillna('0')

df = df.ix[1:]

print(df)

train, test = train_test_split(df, test_size = 0.2)

# MLP trains on two arrays: 
# array X of size (n_samples, n_features), which holds the training samples represented as floating point feature vectors
# array y of size (n_samples,), which holds the target values (class labels) for the training samples
X = train[["feature1", "feature2", "feature3", "feature4"]]
y = train["Outcome"]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)     

X_test = test[["feature1", "feature2", "feature3", "feature4"]]
y_true = test["Outcome"]

# After fitting (training), the model can predict labels for new samples
y_pred = clf.predict(X_test)


print(accuracy_score(y_true, y_pred))


# # Export the result to labeled4test.txt
# text_file = open("result.txt", "w")

# for line in result:
# 	text_file.write(("%s" % line) + "\n")

# text_file.close()