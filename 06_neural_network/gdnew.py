import numpy as np
from neural_network_homemade import NeuralNetwork

class GridSearchCV:

    def __init__(self, parameters, cv=5):
        self.alpha = parameters['alpha']
        self.lmd = parameters['lmd']
        self.epochs = parameters['epochs']
        self.cv = cv

        self.best_score_ = 0
        self.best_params_ = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

    def create_combinations(self):
        for i in self.alpha:
            for k in self.lmd:
                for j in self.epochs:
                    # alpha, lmd, epochs
                    yield i, j, k

    def create_experiments(self, X, y):
        for i in range(self.cv):
            val_indices = list(range(round(i * len(X) / self.cv),
                                     round((i + 1) * len(X) / self.cv)))

            X_train = np.delete(X, val_indices, axis=0)
            y_train = np.delete(y, val_indices, axis=0)
            X_val = X[val_indices]
            y_val = y[val_indices]

            yield X_train, y_train, X_val, y_val

    def fit(self, X, y):
        for alpha, lmd, epochs in self.create_combinations():
            for X_train, y_train, X_val, y_val in self.create_experiments(X, y):
                nn = NeuralNetwork(learning_rate=alpha,
                                   lmd=lmd,
                                   epochs=epochs,
                                   layers=[X_train.shape[1], 6, 6, 1])
                nn.fit(X_train, y_train, X_val, y_val)

                acc = nn.compute_performance(X_val, y_val)['accuracy']
                if acc < self.best_score_:
                    self.best_score_ = acc
                    self.best_params_ = list((alpha, lmd, epochs))
