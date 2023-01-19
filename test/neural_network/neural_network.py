import numpy as np

np.random.seed(42)

class NeuralNetwork():

    def __init__(self, alpha=0.01, lmd=1, epochs=700, layers=[5,5,5,1]):
        self.alpha = alpha
        self.lmd = lmd
        self.epochs = epochs
        self.layers = layers

        self.X = None
        self.y = None

        self.w = {}
        self.b = {}

        self.loss = []
        self.loss_val = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def validate(self, X_val, y_val):
        self.X = X_val
        self.y = y_val
        al = self.forward_propagation()
        cost_val = self.compute_cost(al)
        return  cost_val

    def fit(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.init_parameters()

        for i in range(self.epochs):
            self.X = X
            self.y = y
            al = self.forward_propagation()
            cost = self.compute_cost(al)
            grads = self.back_propagation(al)
            self.update(grads)
            self.loss.append(cost)

            cost_val = self.validate(X_val, y_val)
            self.loss_val.append(cost_val)

    def init_parameters(self):
        L = len(self.layers)
        for l in range (1, L):
            self.w[l] = np.random.randn(self.layers[l], self.layers[l - 1])