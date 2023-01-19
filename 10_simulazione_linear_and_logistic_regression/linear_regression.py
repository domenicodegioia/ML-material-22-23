import numpy as np

np.random.seed(42)

class LinearRegression:

    def __init__(self, learning_rate=0.01, n_steps=2000, n_features=1, lmd=1):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.n_features = n_features
        self.lmd = lmd
        self.lmd_ = np.full((n_features,), lmd)
        self.lmd_[0] = 0
        self.theta = np.random.rand(n_features)

    def fit(self, X, y, X_val, y_val):
        m = X.shape[0]

        c_h = np.zeros(self.n_steps)
        c_h_val = np.zeros(self.n_steps)
        t_h = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = np.dot(X, self.theta)
            preds_val = np.dot(X_val, self.theta)
            error = preds - y
            error_val = preds_val - y_val

            self.theta = self.theta - 1 / m * self.learning_rate * (np.dot(X.T, error) + self.theta.T * self.lmd_)

            c_h[step] = 1 / (2 * m) * np.dot(error.T, error)
            c_h_val[step] = 1 / (2 * m) * np.dot(error_val.T, error_val)
            t_h[step, :] = self.theta.T

        return c_h, c_h_val, t_h

    def predict(self, X):
        X_pred = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X_pred, self.theta)

    def compute_performance(self, X, y):
        preds = self.predict(X)

