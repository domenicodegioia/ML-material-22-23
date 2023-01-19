import numpy as np

np.random.seed(123)


class LogisticRegression():

    def __init__(self, learning_rate=0.01, n_steps=2000, n_features=1, lmd=1):
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.n_features = n_features
        self.lmd = lmd
        self.lmd_ = np.full((n_features), lmd)
        self.lmd_[0] = 0
        self.theta = np.random.rand(n_features)

    def _sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, X_val, y_val):
        m = len(X)

        t_h = np.zeros((self.n_steps, self.theta.shape[0]))
        c_h = np.zeros(self.n_steps)
        c_h_v = np.zeros(self.n_steps)

        for step in range(self.n_steps):
            p = self._sigmoid(np.dot(X, self.theta))
            p_val = self._sigmoid(np.dot(X_val, self.theta))

            e = p - y
            # error_val = preds_val - y_val

            self.theta = self.theta - (1 / m) * self.learning_rate * (np.dot(X.T, e))

            t_h[step, :] = self.theta.T
            c_h[step] = -1 / m * (np.dot(y.T, np.log(p)) + (np.dot((1 - y.T), np.log(p))))
            c_h_v[step] = -1 / m * (np.dot(y_val.T, np.log(p_val)) + (np.dot((1 - y_val.T), np.log(p_val))))

        return t_h, c_h, c_h_v

    def fit_regularized(self, X, y, X_val, y_val):
        m = len(X)

        theta_history = np.zeros((self.n_steps, self.theta.shape[0]))
        cost_history = np.zeros(self.n_steps)
        cost_history_val = np.zeros(self.n_steps)

        for step in range(0, self.n_steps):
            p = self._sigmoid(np.dot(X, self.theta))
            p_val = self._sigmoid(np.dot(X_val, self.theta))

            error = p - y

            self.theta -= ((1 / m) * self.learning_rate * (np.dot(X.T, error) + (self.theta.T * self.lmd_)))

            loss = -(1 / m) * (np.dot(y.T, np.log(p)) + np.dot((1 - y.T), np.log(1 - p)))
            loss_val = -(1 / m) * (np.dot(y_val.T, np.log(p_val)) + np.dot((1 - y_val.T), np.log(1 - p_val)))

            reg = self.lmd / (2 * m) * np.dot(self.theta.T[1:], self.theta[1:])

            theta_history[step, :] = self.theta.T
            cost_history[step] = loss + reg
            cost_history_val[step] = loss_val + reg

        return theta_history, cost_history, cost_history_val

    def predict(self, X, thrs):
        X_preds = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X_preds, self.theta)) >= thrs

    def compute_performance(self, X, y):
        preds = self.predict(X, 0.5)

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