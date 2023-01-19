import math

import numpy as np

np.random.seed(123)


class LinearRegression:

    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1, lmd=1):
        # lmd_ is an array useful when is necessary compute theta's update with regularization factor
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)
        self.lmd = lmd
        self.lmd_ = np.full((n_features,), lmd)
        self.lmd_[0] = 0
        self.n_features = n_features

    def fit_full_batch_gd(self, X, y, X_val, y_val):
        m = X.shape[0]
        m_val = X_val.shape[0]

        c_h = np.zeros(self.n_steps)
        c_h_v = np.zeros(self.n_steps)
        t_h = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            p = np.dot(X, self.theta)
            p_val = np.dot(X_val, self.theta)

            e = p - y
            e_val = p_val - y_val

            self.theta = self.theta - self.learning_rate / m * (np.dot(X.T, e) + self.theta.T * self.lmd_)

            t_h[step, :] = self.theta.T
            c_h[step] = 1 / (2 * m) * (np.dot(e.T, e) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:]))
            c_h_v[step] = 1 / (2 * m_val) * (np.dot(e_val.T, e_val) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:]))

        return c_h, c_h_v, t_h

    def fit_stochastic_gd(self, X, y, X_val, y_val):
        m = X.shape[0]
        m_val = X_val.shape[0]
        n = X.shape[1]

        c_h = np.zeros(self.n_steps)
        c_h_v = np.zeros(self.n_steps)
        t_h = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            for i in range(m):
                p = np.dot(X[i], self.theta)
                e = p - y[i]

                for j in range(n):
                    self.theta[j] = self.theta[j] - self.learning_rate * (
                            np.dot(X[i, j].T, e) + self.theta[j].T * self.lmd
                    )

            p_val = np.dot(X_val, self.theta)
            e_val = p_val - y_val

            t_h[step, :] = self.theta.T
            c_h[step] = 1 / (2 * m) * (np.dot(e.T, e) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:]))
            c_h_v[step] = 1 / (2 * m_val) * (np.dot(e_val.T, e_val) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:]))

        return c_h, c_h_v, t_h

    def fit_mini_batch_gd(self, X, y, X_val, y_val):
        m = X.shape[0]
        m_val = X_val.shape[0]
        n = X.shape[1]

        b = 50
        c_h = np.zeros(self.n_steps)
        c_h_v = np.zeros(self.n_steps)
        t_h = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            for i in range(math.ceil(m/b)):
                p = np.dot(X[i * b:(i + 1) * b], self.theta)
                e = p - y[i * b:(i + 1) * b]

                for j in range(n):
                    self.theta[j] = self.theta[j] - self.learning_rate * (1 / b) * (
                            np.dot(X[i * b:(i + 1) * b, j].T, e) + self.theta[j].T * self.lmd
                    )

            p_val = np.dot(X_val[i * b:(i + 1) * b], self.theta)
            e_val = p_val - y_val[i * b:(i + 1) * b]

            t_h[step, :] = self.theta.T
            c_h[step] = 1 / (2 * m) * (np.dot(e.T, e) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:]))
            c_h_v[step] = 1 / (2 * m_val) * (np.dot(e_val.T, e_val) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:]))

        return c_h, c_h_v, t_h

    def predict(self, X):
        Xpred = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xpred, self.theta)

    def cost_grid(self, X, Y, A, B, first_dim, second_dim):
        result = np.zeros((100, 100))

        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                temp_theta = self.theta[:]
                temp_theta[first_dim] = A[r, c]
                temp_theta[second_dim] = B[r, c]
                result[r, c] = np.average((X @ temp_theta - Y) ** 2) * 0.5

        return result

    def compute_performance(self, X, y):
        preds = self.predict(X)

        mae = self._mean_absolute_error(preds, y)
        mape = self._mean_absolute_percentage_error(preds, y)
        mpe = self._mean_percentage_error(preds, y)
        mse = self._mean_squared_error(preds, y)
        rmse = self._root_mean_squared_error(preds, y)
        r2 = self._r_2(preds, y)

        return {'mae': mae, 'mape': mape, 'mpe': mpe, 'mse': mse, 'rmse': rmse, 'r2': r2}

    def _mean_absolute_error(self, pred, y):
        output_errors = np.abs(pred - y)
        return np.average(output_errors)

    def _mean_squared_error(self, pred, y):
        output_errors = (pred - y) ** 2
        return np.average(output_errors)

    def _root_mean_squared_error(self, pred, y):
        return np.sqrt(self._mean_squared_error(pred, y))

    def _mean_absolute_percentage_error(self, pred, y):
        output_errors = np.abs((pred - y) / y)
        return np.average(output_errors)

    def _mean_percentage_error(self, pred, y):
        output_errors = (pred - y) / y
        return np.average(output_errors) * 100

    def _r_2(self, pred, y):
        sst = np.sum((y - y.mean()) ** 2)
        ssr = np.sum((pred - y) ** 2)

        r2 = 1 - (ssr / sst)
        return r2

    def learning_curves(self, X, y, X_val, y_val):
        m = len(X)
        cost_history = np.zeros(m)
        cost_history_val = np.zeros(m)

        for i in range(m):
            c_h, c_h_v, _ = self.fit_full_batch_gd(X[:i + 1], y[:i + 1], X_val, y_val)
            cost_history[i] = c_h[-1]
            cost_history_val[i] = c_h_v[-1]

        return cost_history, cost_history_val
