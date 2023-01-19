import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit


class NeuralNetwork():
    def __init__(self, learning_rate=0.01, epochs=1000, lmd=1, layers=[5, 5, 5, 1]):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lmd = lmd
        self.layers = layers  # input + hidden + output
        self.w = {}  # weights
        self.b = {}  # bias
        # self.A = {}  # a^(i) = g(z^(i))
        # self.Z = {}  # z^(i) = w[i-1] * a^(i-1)
        # self.dA = {}
        # self.dZ = {}
        self.loss = []
        self.loss_val = []
        self.X = None
        self.y = None

    def get_params(self):
        return {
            'learning_rate': self.learning_rate,
            'lmd': self.lmd,
            'epochs': self.epochs
        }

    def set_params(self, param: dict):
        valid_params = self.get_params()
        for key, values in param.items():
            if key not in valid_params.keys():
                raise Exception(f"No such {key}")
            setattr(self, key, values)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def validate(self, X_val, y_val):
        self.X = X_val
        self.y = y_val
        al = self.forward_propagation()
        cost_val = self.compute_cost(al)
        return cost_val

    def fit(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.init_params()  # initialize weights and bias values

        for _ in range(self.epochs):
            self.X = X
            self.y = y
            # compute z^(i) e a^(i) for each layer starting from X values
            A_list = self.forward_propagation()
            # compute J(theta) for training set
            cost = self.compute_cost(A_list)
            # compute gradient (updates)
            grads = self.back_propagation(A_list)
            # update parameters
            self.update(grads)
            # update loss curve for training set
            self.loss.append(cost)

            # compute J(theta) for validation set
            cost_val = self.validate(X_val, y_val)
            # update loss curve for validation set
            self.loss_val.append(cost_val)

    def init_params(self):
        L = len(self.layers)  # number of layers

        for i in range(1, L):
            self.w[i] = np.random.randn(self.layers[i], self.layers[i - 1])
            self.b[i] = np.zeros((self.layers[i], 1))

    def forward_propagation(self):
        layers = len(self.w)

        values = {}

        for i in range(1, layers + 1):
            if i == 1:
                # for the first layer
                # z(2) = w(1) * x and add bias
                values['Z' + str(i)] = np.dot(self.w[i], self.X.T) + self.b[i]
                # a(2) = g(z(2))
                values['A' + str(i)] = self.sigmoid(values['Z' + str(i)])
            else:
                # for the other layers
                # z(2) = w(1) * a(1) and add bias
                values['Z' + str(i)] = np.dot(self.w[i], values['A' + str(i - 1)]) + self.b[i]
                # a(i) = g(z(i))
                values['A' + str(i)] = self.sigmoid(values['Z' + str(i)])
        return values

    def compute_cost(self, AL):
        layers = len(AL) // 2  # AL contains A and Z elements
        Y_pred = AL['A' + str(layers)]  # last layer of the NN (prediction on training set)
        m = self.y.shape[0]

        # J(theta) pt.1
        cost = -np.average(self.y.T * np.log(Y_pred) + (1 - self.y.T) * np.log(1 - Y_pred))

        reg = 0
        for i in range(1, layers):
            reg += np.sum(np.square(self.w[i]))
        reg2 = (self.lmd / (2 * m)) * reg  # J(theta) pt.2
        return cost + reg2  # J(theta)

    def compute_cost_derivative(self, AL):
        # derivative of J(theta) w.r.t. a (without regularization term) - dJ/dA in the output layer
        return -(np.divide(self.y.T, AL) - np.divide(1 - self.y.T, 1 - AL))

    def back_propagation(self, values):
        m = self.X.shape[0]
        upd = {}
        layers = len(self.w)

        for i in range(layers, 0, -1):
            # compute dJ/dA
            if i == layers:
                # considering A(L) as predictions (h_theta)
                dA = self.compute_cost_derivative(values['A' + str(i)])
                # dJ/dZ = (dJ/dA) * (dA/dZ) = (dJ/dA) * g_prime(Z)
                dZ = np.multiply(dA, self.sigmoid_derivative(values['A' + str(i)]))
            else:
                # dJ/dA(i) = dJ/dZ(i+1) * dZ(i+1)/dA(i) = dJ/dZ(i+1) * W(i) * dA(i)/dA(i) = dJ/dZ(i+1) * W(i)
                dA = np.dot(self.w[i + 1].T, dZ)
                # dJ/dZ = (dJ/dA) * (dA/dZ) = (dJ/dA) * g_prime(Z)
                dZ = np.multiply(dA, self.sigmoid_derivative(values['A' + str(i)]))
            # compute dJ/dW = dJ/dZ * dZ/dW and dJ/dB = dJ/dZ * dZ/dB
            if i == 1:
                upd['W' + str(i)] = (1 / m) * (np.dot(dZ, self.X) + self.lmd * self.w[i])
                upd['B' + str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            else:
                upd['W' + str(i)] = (1 / m) * (np.dot(dZ, values['A' + str(i - 1)].T) + self.lmd * self.w[i])
                upd['B' + str(i)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        return upd

    def update(self, upd):
        layers = len(self.w)
        for i in range(1, layers + 1):
            self.w[i] = self.w[i] - self.learning_rate * upd['W' + str(i)]
            self.b[i] = self.b[i] - self.learning_rate * upd['B' + str(i)]

    def predict(self, X_pred):
        # forward propagation starting from X_pred
        self.X = X_pred
        Al = self.forward_propagation()
        layers = len(Al) // 2
        # last layer (output layer) contains predictions
        pred = Al['A' + str(layers)]
        return np.round(pred)

    def loss_curve(self):
        plt.plot(self.loss, color='b')
        plt.plot(self.loss_val, color='r')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Loss Curve')
        plt.show()

    def compute_performance(self, X, y):
        preds = self.predict(X)

        cm = self._confusion_matrix(y, preds)

        accuracy = self._accuracy(cm)
        precision = self._precision(cm)
        recall = self._recall(cm)
        specificity = self._specificity(cm)
        errore_rate = self._error_rate(cm)
        f_measure = self._f_measure(cm)
        false_positive_rate = self._false_positive_rate(cm)

        return {'confusion matrix': cm,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'errore_rate': errore_rate,
                'f_measure': f_measure,
                'false_positive_rate': false_positive_rate}

    def _confusion_matrix(self, y, preds):
        tp = tn = fp = fn = 0

        v = zip(y,preds)

        for a, p in zip(y, preds):
            if p == a:
                if p == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if p == 1:
                    fp += 1
                else:
                    fn += 1

        # confusion_matrix
        #       0   1
        #   0   tn  fp
        #   1   fn  tp

        return np.array([[tn, fp], [fn, tp]])

    def _accuracy(self, cm):
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        return (tp + tn) / (tp + fn + fp + tn)

    def _precision(self, cm):
        fp = cm[0][1]
        tp = cm[1][1]
        return tp / (tp + fp)

    def _recall(self, cm):
        fn = cm[1][0]
        tp = cm[1][1]
        return tp / (tp + fn)

    def _specificity(self, cm):
        tn = cm[0][0]
        fp = cm[0][1]
        return tn / (fp + tn)

    def _error_rate(self, cm):
        return 1 - self._accuracy(cm)

    def _f_measure(self, cm):
        precision = self._precision(cm)
        recall = self._recall(cm)
        return 2 * precision * recall / (precision + recall)

    def _false_positive_rate(self, cm):
        return 1 - self._specificity(cm)
