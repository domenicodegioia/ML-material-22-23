import numpy as np

np.random.seed(123)

''' dal main la richimiamo come:
    nn = NeuralNet(layers=[X_train.shape[1], 25, 1], learning_rate=0.5, iterations=1000, lmd=0)
    nn.fit(X_train, y_label_train)
    y_hat = nn.predict(X_test)
    nn.rmse(y_label_test, y_hat)
    '''

class NeuralNetworkClass:
    def __init__(self, layers, learning_rate, iterations, lmd):
        self.layers = layers
        self.learning_rate = learning_rate
        self.n_iterations = iterations
        self.lmd = lmd
        self.w = {}
        self.b = {}
        self.loss = []
        self.X = None
        self.y = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def init_weights(self):
        L = len(self.layers)
        np.random.seed(42)

        for l in range(1, L):
            self.w[l] = np.random.randn(self.layers[l], self.layers[l - 1])
            self.b[l] = np.random.randn(self.layers[l], 1)

    def forward_propagation(self):
        L = len(self.layers)
        Z = {}
        A = {0: self.X.T}

        for l in range(1, L):
            Z[l] = np.dot(self.w[l], A[l - 1]) + self.b[l]
            A[l] = self.sigmoid(Z[l])

        return Z, A

    def back_propagation(self, Z, A):
        L = len(self.layers)
        m = len(self.y)

        dW = {}
        dB = {}

        for l in range(L - 1, 0, -1):
            if l == L - 1:
                # -y/a + (1-y)/(1-a) * a(1-a) si semplifica in a - y
                # -y +ya a -ay / a(1-a)
                dZ = A[l] - self.y.T
            else:
                dA = np.dot(self.w[l + 1].T, dZ)
                dZ = np.multiply(dA, self.sigmoid_derivative(A[l]))

            dW[l] = 1 / m * np.dot(dZ, A[l - 1].T) + self.lmd * self.w[l]
            dB[l] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        return dW, dB

    def update_params(self, dW, dB):
        L = len(self.layers)

        for l in range(1, L):
            self.w[l] -= self.learning_rate * dW[l]
            self.b[l] -= self.learning_rate * dB[l]

    def compute_cost(self, A):
        m = len(self.y)
        L = len(self.layers)

        preds = A[len(A) - 1]

        cost = -np.average(self.y.T * np.log(preds) + (1 - self.y.T) * np.log(1 - preds))
        reg_sum = 0
        for l in range(1, len(self.layers)):
            reg_sum += (np.sum(np.square(self.w[l])))
        L2_regularization_cost = reg_sum * (self.lmd / (2 * m))
        return cost + L2_regularization_cost

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.init_weights()

        for i in range(self.n_iterations):
            Z, A = self.forward_propagation()
            dW, dB = self.back_propagation(Z, A)
            self.update_params(dW, dB)
            cost = self.compute_cost(A)
            self.loss.append(cost)

    def predict(self, X, t=0.5):
        self.X = X
        _, A = self.forward_propagation()
        preds = A[len(A) - 1]
        return preds >= t


    def confusion_matrix(self, ytest,  predicted):
        tp, fp, tn, fn = 0, 0, 0, 0
        i = 0
        for pred in predicted:
            if pred == ytest[i]:
                tp += 1
            elif pred == 1 and ytest[i] == 0:
                fp += 1
            elif pred == 0 and ytest[i] == 0:
                tn += 1
            elif pred == 0 and ytest[i] == 1:
                fn += 1
            i += 1
        # sensityvity = recall = tp / tot positivi in y
        recall = tp / (tp + fn)

        # accuracy true / tutto
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # precision = positivi azzeccati / tot positivi predetti
        precision = tp / (tp + fp)

        # specificity = tn rate = negativi azzeccati / negativi reali
        specificity = tn / (tn + fp)

        # error rate = falsi / tutto
        errorrate = (fp + fn) / (tp + fp + tn + fn)

        # fmeasure = 2 * (precision * recall ) / (precision + recall)
        fmeasure = 2 * (precision * recall) / (precision + recall)

        return recall, accuracy, precision, specificity, errorrate, fmeasure