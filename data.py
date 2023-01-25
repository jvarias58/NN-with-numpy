import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape  #m=42000, n=785

test = data[0:1000].T
y_test = test[0]
x_test = test[1:n]


train = data[1000:m].T
y_train = train[0]
x_train = train[1:n]

labels = np.zeros((y_train.size, 10))
for i in range(0,y_train.size):
    labels[i][y_train[i]] = 1
labels = labels.T

labels2 = np.zeros((y_test.size, 10))
for i in range(0,y_test.size):
    labels2[i][y_test[i]] = 1
labels2 = labels2.T