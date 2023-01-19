import pandas as pd
import numpy as np
from KFoldsCV_homemade import KFoldsCrossValidation

SEED = 42

houses = pd.read_csv('../data/houses.csv')
houses = houses.dropna(axis=1)

houses = houses.sample(frac=1, random_state=SEED).reset_index(drop=True)

x = houses[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values
y = houses['SalePrice'].values

train_index = round(len(x) * 0.8)

X_train = x[:train_index]
y_train = y[:train_index]

X_test = x[train_index:]
y_test = y[train_index:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # add bias column

# Perform a K-folds CV with k = 5 (default)
k_f = KFoldsCrossValidation(X_train, y_train)
errors, thetas, best_index = k_f.validate()

print(f"Thetas: {thetas}")
print(f"MSE: {errors}")
print(f"The best Theta values are: {thetas[best_index]}")
print(f"The best MSE is: {errors.min()}")
print(f"The mean of the errors is equal to: {errors.mean()}")
