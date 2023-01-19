import numpy as np


class LinearRegression:

    def __init__(self, lmd=1, alpha=0.01, epochs=2000, n=1):
        self.lmd = lmd
        self.lmd_ = np.full((n,), lmd)
        self.lmd_[0] = 0
        self.alpha = alpha
        self.epochs = epochs
        self.n = n
        self.theta = np.random.rand(n)

    def fit_full_batch_gd(self, X, y, X_val, y_val):
        m = X.shape[0]
        m_val = X_val.shape[0]

        t_h = np.zeros((self.epochs, self.theta.shape[0]))
        c_h = np.zeros(self.epochs)
        c_h_v = np.zeros(self.epochs)

        for step in range(0, self.epochs):
            p = np.dot(X, self.theta)
            e = p - y
            p_val = np.dot(X_val, self.theta)
            e_val = p_val - y_val

            self.theta -= self.alpha / m * (
                np.dot(X.T, e) + self.theta.T * self.lmd_
            )

            t_h[step, :] = self.theta.T
            c_h[step] = 1 / (2 * m) * (
                np.dot(e.T, e) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:])
            )
            c_h_v[step] = 1 / (2 * m_val) * (
                np.dot(e_val.T, e_val) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:])
            )

        return t_h, c_h, c_h_v

    def fit_stochastic_gd(self, X, y, X_val, y_val):
        m = X.shape[0]
        m_val = X_val.shape[0]

        t_h = np.zeros((self.epochs, self.theta.shape[0]))
        c_h = np.zeros(self.epochs)
        c_h_v = np.zeros(self.epochs)

        for step in range(0, self.epochs):
            e = np.zeros(m)
            p = np.zeros(m)

            for i in range(m):
                p[i] = np.dot(X[i], self.theta)
                e[i] = p[i] - y[i]

                for j in range(self.n):
                    self.theta[j] = self.theta[j] - self.alpha * (
                        np.dot(X[i,j].T, e[i]) + self.theta.T * self.lmd
                    )

            p_val = np.dot(X_val, self.theta)
            e_val = p_val - y_val

            t_h[step, :] = self.theta.T
            c_h[step] = 1 / (2 * m) * (
                np.dot(e.T, e) + self.lmd * np.dot(self.theta[1:,j].T, self.theta[1:,j])
            )
            c_h_v[step] = 1 / (2 * m_val) * (
                    np.dot(e_val.T, e_val) + self.lmd * np.dot(self.theta[1:, j].T, self.theta[1:, j])
            )

        return t_h, c_h, c_h_v

    def fit_mini_batch_gd(self, X, y, X_val, y_val):
        m = X.shape[0]
        m_val = X_val.shape[0]
        b = 100

        t_h = np.zeros((self.epochs, self.theta.shape[0]))
        c_h = np.zeros(self.epochs)
        c_h_v = np.zeros(self.epochs)

        for step in range(0, self.epochs):
            for i in range(m):
                p = np.dot(X[i*b:(i+1)*b], self.theta)
                e = p - y[i]

                for j in range(self.n):
                    self.theta[j] -= self.alpha * (
                        np.dot(X[i,j].T, e) + self.theta[j].T * self.lmd
                    )

            p_val = np.dot(X_val, self.theta)
            e_val = p_val - y_val

            t_h[step, :] = self.theta.T
            c_h[step] = 1 / (2 * m) * (
                np.dot(e.T, e) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:])
            )
            c_h_v[step] = 1 / (2 * m_val) * (
                    np.dot(e_val.T, e_val) + self.lmd * np.dot(self.theta[1:].T, self.theta[1:])
            )

        return t_h, c_h, c_h_v

    def predict(self, X):
        X_pred = np.c_[np.ones(X.shape[0]), X]
        pred = np.dot(X_pred, self.theta)
        return pred

    def compute_performance(self, X, y):
        pred = self.predict(X)

        mae = self._mae(pred, y)
        mse = self._mse(pred, y)
        rmse = self._rmse(pred, y)
        mape = self._mape(pred, y)
        mpe = self._mpe(pred, y)
        r2 = self._r2(pred, y)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mpe': mpe,
            'r2': r2,
        }

    def _mae(self, pred, y):
        return np.average(np.abs(pred - y))

    def _mse(self, pred, y):
        return np.average((pred - y) ** 2)

    def _rmse(self, pred, y):
        return np.sqrt(self._mse(pred, y))

    def _mape(self, pred, y):
        return np.average(np.abs((pred - y) / y))

    def _mpe(self, pred, y):
        return np.average((pred - y) / y * 100)

    def _r2(self, pred, y):
        ssr = np.sum((y - y.mean()) ** 2)
        sst = np.sum((pred - y) ** 2)
        r2 = 1 - ssr / sst
        return r2

