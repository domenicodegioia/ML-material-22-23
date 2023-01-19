import numpy as np

np.random.seed(123)

class LinearRegression:
    def __init__(self, nfeatures, steps=1000, a=0.05, lmd=2):
        self.steps = steps
        self.a = a
        self.lmd = lmd
        self.lmd_ = np.full(nfeatures, lmd)
        self.lmd_[0] = 0
        self.thetas = np.random.randn(nfeatures)

    def fit_no_reg(self, X, y, X_val, y_val):

        m = len(X)
        cost_history = np.zeros(self.n_steps)
        cost_history_val = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = np.dot(X, self.theta)
            preds_val = np.dot(X_val, self.theta)

            error = preds - y
            error_val = preds_val - y_val

            self.theta = self.theta - (self.learning_rate * (1/m) * np.dot(X.T, error))
            theta_history[step, :] = self.theta.T
            cost_history[step] = 1/(2*m) * np.dot(error.T, error)
            cost_history_val[step] = 1 / (2 * m) * np.dot(error_val.T, error_val)

        return cost_history, cost_history_val, theta_history

    def fit(self, x, xva, y, yva):
        m = len(x)
        cost_history = np.zeros(self.steps)
        cost_history_va = np.zeros(self.steps)
        theta_history = np.zeros((self.steps, self.thetas.shape[0]))

        for step in range(self.steps):
            pred = np.dot(x, self.thetas)
            pred_va = np.dot(xva, self.thetas)
            error = pred - y
            err_va = pred_va - yva

            self.thetas = self.thetas - ((self.a / m) * (np.dot(x.T, error) + (self.thetas * self.lmd_)))
            theta_history[step] = self.thetas
            cost_history_va = (1 / (2 * m)) * (np.dot(err_va.T, err_va) +
                                               (self.lmd * np.dot(self.thetas.T, self.thetas)))
            cost_history[step] = (1 / (2 * m)) * (np.dot(error.T, error) +
                                                  (self.lmd * np.dot(self.thetas.T[1:], self.thetas[1:])))

        return cost_history, cost_history_va, theta_history

    def fit_stochastic(self, x, y, xva, yva):
        cost_history = np.zeros(self.steps)
        cost_history_va = np.zeros(self.steps)
        thetas_history = np.zeros(self.steps, (self.thetas.shape[0]))
        m = len(x)

        for step in range(self.steps):
            pred_va = np.dot(xva, self.thetas)
            error_va = pred_va - yva
            cost = 0
            for i in range(m):
                x_i = x[i, :]
                y_i = y[i]
                pred = np.dot(x_i, self.thetas)
                error = pred - y_i
                self.thetas = self.thetas - self.a * ( np.dot(x_i.T, error) + np.dot(self.lmd_.T, self.thetas))
                cost += 0.5 * (np.dot(error.T, error) + (self.lmd * np.dot(self.thetas.T, self.thetas)))

            cost_history_va[step] = (1 / (2 * m)) * (np.dot(error_va.T, error_va) + self.lmd * np.dot(self.thetas.T[1:], self.thetas[1:]))
            cost_history[step] = (1 / m) * cost
            thetas_history[step] = self.thetas
        return cost_history, cost_history_va, thetas_history

    def fit_minitbatch(self, x, y, xval, yval, batch_size = 100):
        cost_history = np.zeros(self.steps)
        cost_history_va = np.zeros(self.steps)
        thetas_history = np.zeros(self.steps, (self.thetas.shape[0]))
        m = len(x)

        for step in range(self.steps):
            pred_va = np.dot(xval, self.thetas)
            error_va = pred_va - yval
            cost = 0
            for i in range(0, m, batch_size):
                x_i = x[i: i+batch_size]
                y_i = y[i: i+batch_size]
                pred = np.dot(x_i, self.thetas)
                error = pred - y_i
                self.thetas = self.thetas - self.a * (np.dot(x_i.T, error) + np.dot(self.lmd_.T, self.thetas))
                cost += 0.5 * (np.dot(error.T, error) + (self.lmd * np.dot(self.thetas.T, self.thetas)))
            cost_history_va[step] = (1 / (2 * m)) * (
                        np.dot(error_va.T, error_va) + self.lmd * np.dot(self.thetas.T[1:], self.thetas[1:]))
            cost_history[step] = (1 / m) * cost
            thetas_history[step] = self.thetas
        return cost_history, cost_history_va, thetas_history

    def predict(self, x):
        x = np.c_[np.ones(x.shape[0]), x]
        return np.dot(x, self.thetas)

    def curve(self, x, y, xva, yva):
        m = len(x)
        cost_history = np.zeros(m)
        cost_history_va = np.zeros(m)

        for i in range(m):
            x_i = x[:i+1]
            y_i = y[:i+1]
            c_h, cv, _ = self.fit(x, y, xva, yva)
            cost_history[i] = c_h[-1]
            cost_history_va[i] = cv[-1]

    def metrics(self, x, y):
        prev = self.predict(x)
        mae = self.mae(prev, y)
        mse = self.mse(prev, y)
        mpe = self.mpe(prev, y)
        mape = self.mape(prev, y)
        r2 = self.r2(prev, y)
        print("MAE: {}  MSE: {}  MPE: {}  mape: {}  R2: {}".format(mae, mse, mpe, mape, r2))


    def mae(self, pred, y):
        return np.average(np.abs(pred - y))

    def mse(self, pred, y):
        square = (pred - y) ** 2
        return np.average(square)

    def rmse(self, pred, y):
        return np.sqrt(self.mse(pred, y))

    def mpe(self, pred, y):
        err = (pred - y)/ y
        return np.average(err)

    def mape(self, pred, y):
        err = np.abs((pred - y)/ y)
        return np.average(err)

    def r2(self, pred, y ):
        a = np.sum((pred - y) ** 2)
        b = np.sum((y - y.mean()) ** 2)
        return 1 - (a / b)

