import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

from logistic_regression_homemade import LogisticRegression

plt.style.use(['ggplot'])

warnings.filterwarnings('ignore')

SEED = 42

bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['diagnosis'] = bc.target

df = df.sample(frac=1).reset_index(drop=True)

X = df.iloc[::-1].values
y = df['diagnosis'].values

train_i = round(len(X) * 0.8)

val_i = round(train_i * 0.7)

X_train = X[:train_i]
y_train = y[:train_i]

X_test = X[train_i:]
y_test = y[train_i:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = np.c_[np.ones(X_train.shape[0]), X_train]

X_val = X_train[val_i:]
y_val = y_train[val_i:]

X_train = X_train[:val_i]
y_train = y_train[:val_i]

log_reg = LogisticRegression(n_features=X_train.shape[1],
                             n_steps=1000,
                             learning_rate=0.01,
                             lmd=10)

cost_history, cost_history_val, theta_history = log_reg.fit(X_train, y_train, X_val, y_val)

print(f'''Thetas: {*log_reg.theta,}''')

res = log_reg.compute_performance(X_test, y_test)
print('\nperformance:\n')
print(''.join(['%s \t= %s\n' % (key, value) for (key, value) in res.items()]))
