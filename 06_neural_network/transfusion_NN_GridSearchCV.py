import pandas as pd
import numpy as np

from neural_network_homemade import NeuralNetwork

np.random.seed(42)

transfusion_df = pd.read_csv('../data/transfusion.csv')
print(transfusion_df.columns)

transfusion_df.rename(columns={"whether he/she donated blood in March 2007": "donatedblood"}, inplace=True)

transfusion_df['donateblood'] = transfusion_df['donatedblood'].replace('y', 1)

# shuffle to avoid group bias
index = transfusion_df.index
transfusion_df = transfusion_df.iloc[np.random.choice(index, len(index))]

X = transfusion_df.drop(['donateblood'], axis=1).values
y = transfusion_df['donateblood'].values

train_index = round(len(X) * 0.8)
val_index = round(train_index * 0.7)

X_train = X[:train_index]
y_train = y[:train_index]

X_test = X[train_index:]
y_test = y[train_index:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_val = X_train[val_index:]
y_val = y_train[val_index:]

X_train = X_train[:val_index]
y_train = y_train[:val_index]

nn = NeuralNetwork(learning_rate=0.01, lmd=1, epochs=5500, layers=[X_train.shape[1], 5, 5, 1])
nn.fit(X_train, y_train, X_val, y_val)
nn.loss_curve()

from gdnew import GridSearchCV

parameters = {
    "alpha": [0.1, 0.01, 0.001],
    "lmd": [1, 0.5, 0.1, 0.01],
    "epochs": [500, 1000, 1500]
}

clf = GridSearchCV(parameters=parameters,
                   cv=5)

clf.fit(X_train, y_train)

print("Best params: ", clf.best_params_)
print("Best score: ", clf.best_score_)

nn.set_params(clf.best_params_)
nn.fit(clf.X_train, clf.y_train, X_val, y_val)
nn.loss_curve()
