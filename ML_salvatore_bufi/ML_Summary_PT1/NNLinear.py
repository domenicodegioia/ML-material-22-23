import numpy as np

np.random.seed(123)

''' dal main la richimiamo come:
    nn = NeuralNet(layers=[X_train.shape[1], 25, 1], learning_rate=0.5, iterations=1000, lmd=0)
    nn.fit(X_train, y_label_train)
    y_hat = nn.predict(X_test)
    nn.rmse(y_label_test, y_hat)
    '''


class NeuralLinear:

    def __init__(self, layers=[13, 8, 2], learning_rate=0.05, steps= 1000, lmd=1):
        self.layers = layers
        self.learning_rate = learning_rate
        self.steps = steps
        self.lmd = lmd
        self.w = {}
        self.b = {}
        self.Y = None
        self.X = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def init_weights(self):
        L = len(self.layers)
        for l in range(1, L):
            self.w[l] = np.random.randn(self.layers[l], self.layers[l-1])
            self.b[l] = np.random.randn(self.layers[l], 1)

    def forward_propagation(self):
        L = len(self.layers)
        A = {0: self.X.T}
        Z = {}

        for l in range(1, L):
            Z[l] = np.dot(self.w[l], A[l-1]) + self.b[l]
            if l == L - 1:
                A[l] = Z[l]
            else:
                A[l] = self.sigmoid(Z[l])
        return Z, A

    def back_propagation(self, Z, A):
        L = len(self.layers)
        dW = {}
        dB = {}
        m = len(self.X)
        for l in range(L-1, 0, -1):
            if l == L - 1:
                dA = A[l] - self.Y.T
                dZ = dA
            else:
                dA = np.dot(self.w[l+1].T, dZ)
                dZ = np.multiply(dA, self.sigmoid_derivative(A[l]))
            dW[l] = 1/ m * np.dot(dZ, A[l-1].T) + (self.lmd * self.w[l])
            dB[l] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        return dW, dB

    def update_param(self, dW, dB):
        L = len(self.layers)
        for l in range(1, L):
            self.w[l] -= self.learning_rate * dW[l]
            self.b[l] -= self.learning_rate * dB[l]

    def cost(self, A):
        m = len(self.Y.T)
        L = len(self.layers)
        cost = (1 / (2 * m)) * np.sum(np.square(A[L-1] - self.Y.T))
        reg = 0
        for l in range(1, L):
            reg += (np.sum(np.square(self.w[l])))
        reg_cost = reg * ( self.lmd / (2 * m))
        return cost + reg_cost

    def predict(self, X):
        self.X = X
        _, A = self.forward_propagation()
        L = len(self.layers)
        prediction = A[L-1]
        return prediction

    def fit(self, x, y):
        self.X = x
        self.Y = y
        cost = []
        self.init_weights()
        for step in range(self.steps):
            Z, A = self.forward_propagation()
            dW, dB = self.back_propagation(Z, A)
            self.update_param(dW, dB)
            costo = self.cost(A)
            cost.append(costo)

    def rmse2(self, pred, y):
        n = len(y)
        square = np.sqrt(np.average((pred - y) ** 2))

        return np.sqrt(square)

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