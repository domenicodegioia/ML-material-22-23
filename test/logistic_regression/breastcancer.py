import pandas as pd
from sklearn.datasets import load_breast_cancer
import numpy as np

from logistic_regression import LogisticRegression

cancer = pd.read_csv('cancer.csv')

cancer = cancer.drop('id', axis=1)

cancer['diagnosis'] = cancer['diagnosis'].replace('B', 0).replace('M', 1)

X = cancer.drop('diagnosis', axis=1).values
y = cancer['diagnosis'].values

train_i = round(len(X) * 0.8)

X_train = X[:train_i]
y_train = y[:train_i]

X_test = X[train_i:]
y_test = y[train_i:]

# mean = X_train.mean(axis=1)
# std = X_train.std(axis=1)
#
# X_train = (X_train - mean) / std
# X_test = (X_test - mean) / std

a = 0
b = 1
min = X_train.min(axis=0)
max = X_train.max(axis=0)
X_train = (X_train - min)/(max - min) * (b - a) + a
X_test = (X_test - min)/(max - min) * (b - a) + a

X_train = np.c_[np.ones(X_train.shape[0]), X_train]

val_index = round(train_i * 0.7)

X_val = X_train[val_index:]
y_val = y_train[val_index:]

X_train = X_train[:val_index]
y_train = y_train[:val_index]

log_reg = LogisticRegression(lmd=1, alpha=0.01, epochs=1500, n=X_train.shape[1])

c_h, c_h_v, t_h = log_reg.fit(X_train, y_train, X_val, y_val)

print('Theta:\t', {*log_reg.theta})
print('Final training cost:\t', c_h[-1])
print('Final validation cost:\t', c_h_v[-1])
