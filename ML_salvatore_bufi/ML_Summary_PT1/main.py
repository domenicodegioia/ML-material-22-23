import numpy as np
import pandas as pd
from LinearRegression import LinearRegression
from LogisticRegression import LogisticRegression
np.random.seed(123)

# IMPORT DATASET
houses = pd.read_csv('./houses.csv')
# se non ha header
wine = pd.read_csv('../data/wine.csv', header=None)


# SHUFFLING, AVOID GRUP BIAS, frac=1 == 100%, reset_index drop= sostituiamo indici con numerici
houses = houses.sample(frac=1).reset_index(drop=True)


# COMBINATION OF FEATURES ES
houses['average_rating'] = houses[['OverallQual', 'OverallCond']].mean(axis=1)


# REPLACE OF FEATURES , es true/false con 0 e 1
wine['deasese'] = wine['desease'].replace('False', 0)
wine['deasese'] = wine['desease'].replace('True', 1)

# SELECT FEATURES
# opzione 1: per nome di colonne
x = houses[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values
y = houses['SalePrice'].values

# opzione 2A: Trasformo in np.array e faccio slicing
y = houses.values[:, -1]
x = houses.values[:, 0:10]

# opzione 2B: iloc
x = houses.drop(houses.iloc[:, 0:10], axis=1).values
x = houses.drop(houses.iloc[:, [1, 2, 3]], axis=1).values


# HOLD OUT SPLITTING = 80% train , 20%test
train_index = round(len(x) * 0.8)
X_train = x[:train_index]
y_train = y[:train_index]
X_test = x[train_index:]
y_test = y[train_index:]

# HOLD OUT STRATIFICATO, supponiamo che feature da predire sia a colonna 0
data = houses.values
target1 = []
target2 = []

for i in range(len(data)):
    if data[i, 0] == 1:
        target1.append(data[i])
    else:
        target2.append(data[i])

target1 = np.array(target1)
target2 = np.array(target2)
train_index_1 = round(len(target1) * 0.8)
train_index_2 = round(len(target2) * 0.8)

data_train = np.concatenate((target1[:train_index_1], target2[:train_index_2]), axis=0)
data_test = np.concatenate((target1[train_index_1:], target2[train_index_2:]), axis=0)

np.random.shuffle(data_test)
np.random.shuffle(data_train)

x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]


# ZSCORE NORMALIZATION, axis = 0 (verticale) | , axis=1 orrizzontale ->
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# MIN/MAX NORMALIZATION es tra 0 e 100
a = 0
b = 100
x_train = ((x_train - min(x_train)) / (max(x_train) - min(x_train))) * (b - a) + a


# COLONNA BIAS PE RI THETA0 ( valore che non viene moltiplicato per gli xi)
# x.shape == (righe, colonne) ==> x.shape[0] = righe, x.shape[1] = colonne
# np.c_ .. aggiungiamo matrice di dimensione righeX x 1, composta da tutti 1 davanti a X_train
# np.ones([4,2]) = mat 1 4 righe 2 colonne
X_train = np.c_[np.ones(X_train.shape[0]), X_train]


# SPLIT TRAINING SET into training and validation ( 70% train , 30% val)
validation_index = round(train_index * 0.7)
X_validation = X_train[validation_index:]
y_validation = y_train[validation_index:]

X_train = X_train[:validation_index]
y_train = y_train[:validation_index]


# K FOLD PER SCEGLIERE BEST FEATURES
k = 4
fold = round(len(X_train) / k)

features_list = []
for feature in range(X_train.shape[1]):
    feature_GER = 0
    for j in range(0, k):
        if j == k - 1:
            x_validation = X_train[k*j:, feature]
            y_validation = y_train[k*j:]
            x_train = X_train[0:k*j, feature]
            y_train1 = y_train[0:k*j]

        else:
            x_validation = X_train[k*j:k*(j+1), feature]
            y_validation = y_train[k*j:k*(j+1)]
            x_train = np.concatenate((X_train[k * (j + 1):, feature], X_train[0:k * j, feature]), axis=0)
            y_train1 = np.concatenate((y_train[0:k * j], y_train[k * (j + 1):]), axis=0)

        # x_train_list = X_train[k*(j+1):, feature] + X_train[0:k*j, feature]
        # x_train = np.array(x_train_list)


        # bias column
        x_validation = np.c_[np.ones(x_validation.shape[0]), x_validation]
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]

        # y_train_list = y_train[0:k*j] + y_train[k*(j+1):]
        # y_train1 = np.arrat(y_train)


        regressor = LinearRegression(nfeatures=2, steps=1000, a=0.05, lmd=2)
        _, cost_list, _ = regressor.fit_reg(x_train, y_train1, x_validation, y_validation)
        feature_GER += cost_list[-1]
    feature_GER = feature_GER / k
    features_list.append((feature_GER, feature))

# best for features
features_list.sort(key=lambda x: x[0])
# retrain on the best 4 features
column = []
n_features = 4
for j in [x[1] for x in features_list]:
    if len(column) < n_features:
        column.append(j)

X_train = X_train[:, column]
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_validation = X_train[validation_index:, :]
X_train = X_train[0:validation_index, :]

y_validation = y_train[validation_index:]
y_train = y_train[0:validation_index]


# REGRESSOR E FIT DATI
linear = LinearRegression(nfeatures=X_train.shape[1], steps=1000, a=0.05, lmd=2)
cost_history, cost_history_val, theta_history = linear.fit(X_train, y_train, X_validation, y_validation)


# CLASSIFICAZIONE MULTICLASS
y = wine[['fruity', 'chocholate', 'caramel']].values
# solite robe, normalizzaz  ecc
lr, pred = [], [] # lista di regressori e valori predetti
for i in range(y.shape[1]):
    lr[i] = LogisticRegression(learning_rate=X_train.shape[1], n_steps=1000)
    predicted = lr[i].predict(x_train, 0.6)
    pred[i] = predicted
pred_fruity = pred[0]
pred_choco = pred[1]
pred_caramel = pred[2]



