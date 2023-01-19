import numpy as np

from metrics.regression_metrics import RegressionMetrics

np.random.seed(42)


class OLSLinearRegression:
    def __init__(self, n_features=1, n_steps=2000):
        self.n_features = n_features
        self.n_steps = n_steps
        self.theta = np.random.randn(n_features)

    def fit(self, x_train, y_train):
        theta_history = np.zeros((self.n_steps, self.n_features))

        for step in range(self.n_steps):
            self.theta = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
            theta_history[step] = self.theta.T

        return theta_history

    def predict(self, x):
        x_pred = np.c_[np.ones(x.shape[0]), x]
        return np.dot(x_pred, self.theta)

    def compute_performance(self, x, y):
        pred = self.predict(x)
        metrics = RegressionMetrics(pred, y)
        return metrics.compute_errors()
