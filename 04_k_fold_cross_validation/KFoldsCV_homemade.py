import numpy as np
from linear_regression_homemade import LinearRegression

class KFoldsCrossValidation:
    def __init__(self, X, y, k=5):
        self.X = X
        self.y = y
        self.k = k

    # generator of k folds
    def create_experiments(self):
        for i in range(self.k):
            # lista degli indici del validation set
            val_indices = list(range(round(i * len(self.X) / self.k),
                                     round((i + 1) * len(self.X) / self.k)))

            X_train = np.delete(self.X, val_indices, axis=0)
            y_train = np.delete(self.y, val_indices, axis=0)
            X_val = self.X[val_indices]
            y_val = self.y[val_indices]

            yield i, X_train, y_train, X_val, y_val

    def validate(self):
        errors = []
        thetas = []

        best_index = 0
        best_score = 0

        for i, X_train, y_train, X_val, y_val in self.create_experiments():
            lr = LinearRegression(n_features=X_train.shape[1])
            lr.fit_full_batch_gd(X_train, y_train, X_val, y_val)

            # delete bias column
            X_val = np.delete(X_val, 0, axis=1)

            preds = lr.predict(X_val)

            mse = lr._mean_squared_error(preds, y_val)  # evaluate
            if mse < best_score:  # compare
                best_score = mse
                best_index = i

            errors.append(mse)
            thetas.append(lr.theta)

        return np.array(errors), np.array(thetas), best_index
