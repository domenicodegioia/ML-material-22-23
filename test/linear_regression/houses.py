import pandas as pd
import numpy as np

from linear_regression import LinearRegression

houses = pd.read_csv('houses.csv')

houses = houses.dropna(axis=1)

houses = houses.sample(frac=1).reset_index(drop=True)

X = houses[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values
y = houses['SalePrice'].values

train_i = round(len(X) * 0.8)

X_train = X[:train_i]
y_train = y[:train_i]

X_test = X[train_i:]
y_test = y[train_i:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = np.c_[np.ones(X_train.shape[0]), X_train]

val_i = round(train_i * 0.7)

X_val = X_train[:val_i]
y_val = y_train[:val_i]

X_train = X_train[val_i:]
y_train = y_train[val_i:]

linear_regression = LinearRegression(n=X_train.shape[1],
                                     epochs=1000,
                                     alpha=0.00001,
                                     lmd=0.1)

t_h, c_h, c_h_v = linear_regression.fit_full_batch_gd(X_train, y_train, X_val, y_val)

print('FULL BATCH GD')
print('Thetas:\t', linear_regression.theta)
print('Final training cost:\t', c_h[-1])
print('Final validation cost:\t', c_h_v[-1])

t_h, c_h, c_h_v = linear_regression.fit_stochastic_gd(X_train, y_train, X_val, y_val)

print('STOCHASTIC GD')
print('Thetas:\t', linear_regression.theta)
print('Final training cost:\t', c_h[-1])
print('Final validation cost:\t', c_h_v[-1])

t_h, c_h, c_h_v = linear_regression.fit_mini_batch_gd(X_train, y_train, X_val, y_val)

print('MINI BATCH GD')
print('Thetas:\t', linear_regression.theta)
print('Final training cost:\t', c_h[-1])
print('Final validation cost:\t', c_h_v[-1])