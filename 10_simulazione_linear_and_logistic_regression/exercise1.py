import numpy as np
import pandas as pd

from linear_regression import LinearRegression

insurance = pd.read_csv('../data/insurance.csv')

insurance["sex"] = insurance["sex"].replace("male", 0).replace("female", 1)
insurance["smoker"] = insurance["smoker"].replace("no", 0).replace("yes", 1)

insurance = insurance.drop("region", axis=1)

insurance = insurance.sample(frac=1).reset_index(drop=True)

X = insurance[['age', 'sex', 'bmi', 'children', 'smoker']].values
y = insurance['charges'].values

train_index = round(X.shape[0] * 0.8)
val_index = round(train_index * 0.7)

X_train = X[:train_index]
y_train = y[:train_index]

X_test = X[train_index:]
y_test = y[train_index:]

mean = X.mean(axis=0)
std = X.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = np.c_[np.ones(X_train.shape[0]), X_train]

X_val = X_train[val_index:]
y_val = y_train[val_index:]

X_train = X_train[:val_index]
y_train = y_train[:val_index]

lin_reg = LinearRegression(learning_rate=0.01, n_steps=1000, n_features=X_train.shape[1], lmd=10)

c_h, c_h_val, t_h = lin_reg.fit(X_train, y_train, X_val, y_val)

print(f'''Thetas: {*lin_reg.theta,}''')
print(f'''Final train cost/MSE:  {c_h[-1]:.3f}''')
print(f'''Final validation cost/MSE:  {c_h_val[-1]:.3f}''')