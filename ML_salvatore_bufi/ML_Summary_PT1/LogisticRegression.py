import numpy as np


np.random.seed(123)


class LogisticRegression:

    def __init__(self, learning_rate=1e-2, n_steps=2000, n_features=1, lmd=1):

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.theta = np.random.rand(n_features)
        self.lmd = lmd
        self.lmd_ = np.full((n_features,), lmd)
        self.lmd_[0] = 0

    def _sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def fit(self, X, y, X_val, y_val):

        m = len(X)
        cost_history_train = np.zeros(self.n_steps)
        cost_history_val = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = self._sigmoid(np.dot(X, self.theta))
            preds_val = self._sigmoid(np.dot(X_val, self.theta))

            error = preds - y

            self.theta = self.theta - (self.learning_rate * (1 / m) * (np.dot(X.T, error) + (self.theta.T*self.lmd_)))
            theta_history[step, :] = self.theta.T

            cost_history_train[step] = -(1/m) * (np.dot(y.T, np.log(preds)) + np.dot((1-y.T), np.log(1-preds)))
            cost_history_val[step] = -(1/m) * (np.dot(y_val.T, np.log(preds_val)) + np.dot((1-y_val.T),
                                                                                           np.log(1-preds_val)))

        return cost_history_train, cost_history_val, theta_history

    def fit_reg(self, X, y, X_val, y_val):

        m = len(X)
        cost_history_train = np.zeros(self.n_steps)
        cost_history_val = np.zeros(self.n_steps)
        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))

        for step in range(0, self.n_steps):
            preds = self._sigmoid(np.dot(X, self.theta))
            preds_val = self._sigmoid(np.dot(X_val, self.theta))

            error = preds - y

            self.theta = self.theta - (self.learning_rate * (1 / m) * (np.dot(X.T, error) + (self.theta.T*self.lmd_)))
            theta_history[step, :] = self.theta.T

            loss = -(1/m) * (np.dot(y.T, np.log(preds)) + np.dot((1-y.T), np.log(1-preds)))
            loss_validation = -(1 / m) * (
                        np.dot(y_val.T, np.log(preds_val)) + np.dot((1 - y_val.T), np.log(1 - preds_val)))
            reg = (self.lmd / (2*m)) * np.dot(self.theta.T[1:], self.theta[1:])



            cost_history_train[step] = loss + reg
            cost_history_val[step] = loss_validation + reg

        return cost_history_train, cost_history_val, theta_history

    def _predict_prob(self, X):

        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        """
        perform a complete prediction about X samples
        :param X: test sample with shape (m, n_features)
        :param threshold: threshold value to disambiguate positive or negative sample
        :return: prediction wrt X sample. The shape of return array is (m,)
        """
        Xpred = np.c_[np.ones(X.shape[0]), X]
        return self._predict_prob(Xpred) >= threshold

    def confusion_matrix(self, xtest, treshold, ytest):
        tp, fp, tn, fn = 0, 0, 0, 0
        i = 0
        predicted = self.predict(xtest, treshold)
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
