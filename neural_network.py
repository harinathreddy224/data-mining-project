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

startDay = '1990-01-01'
endDay = '2015-12-31'
stock = Share('^HSI') 
filePath = './data/hsi.csv'

# # Get data and save to csv
history = stock.get_historical(startDay, endDay)
df = pd.DataFrame(history)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date', ascending=False)
df.to_csv(filePath)

# Plot graph
plt.plot(df['Close'])
# pylab.show()

# Add label
# df = pd.read_csv(filePath)
df["label"] = np.where(df['Close'] > df['Open'], '+1', '-1')
# df['ho'] = df['High'] - df['Open'] # distance between Highest and Opening price
# df['lo'] = df['Low'] - df['Open'] # distance between Lowest and Opening price
# df['gain'] = df['Close'] - df['Open']

df['feature1'] = df['Close'] > df['Close'].shift()
df['feature2'] = df['Close'] > df['Close'].shift().shift()
df['feature3'] = df['Close'] > df['Close'].shift().shift().shift()
df['feature4'] = df['Close'] > df['Close'].shift().shift().shift().shift()
df['feature5'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift()
df['feature6'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift()
df['feature7'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift()
df['feature8'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift()
df['feature9'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift().shift()
df['feature10'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift().shift().shift()
df['feature11'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift()
df['feature12'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift()
df['feature13'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift()
df['feature14'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift()
df['feature15'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift()
df['feature16'] = df['Close'] > df['Close'].shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift().shift()

# df['feature1'] = df['Open']
# df['feature2'] = df['Close']
# df['feature3'] = df['Open'].shift()
# df['feature4'] = df['Close'].shift()

df = df.ix[16:]

print(df)

train, test = train_test_split(df, test_size = 0.2)

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


# # Export the result to labeled4test.txt
# text_file = open("result.txt", "w")

# for line in result:
# 	text_file.write(("%s" % line) + "\n")

# text_file.close()