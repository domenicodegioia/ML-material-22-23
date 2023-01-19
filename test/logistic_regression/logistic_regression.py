import numpy as np

class LogisticRegression:

    def __init__(self, lmd, alpha, epochs, n):
        self.lmd = lmd
        self.lmd_ = np.full((n,), lmd)
        self.lmd_[0] = 0
        self.alpha = alpha
        self.epochs = epochs
        self.n = n
        self.theta = np.random.rand(n)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, X_val, y_val):
        m = X.shape[0]
        m_val = X_val.shape[0]

        c_h = np.zeros(self.epochs)
        c_h_v = np.zeros(self.epochs)
        t_h = np.zeros((self.epochs, self.theta.shape[0]))

        for step in range(self.epochs):
            p = self._sigmoid(np.dot(X, self.theta))
            e = p - y
            p_val = self._sigmoid(np.dot(X_val, self.theta))
            # e_val = p_val - y_val

            self.theta -= self.alpha / m * (
                np.dot(X.T, e) + self.theta.T * self.lmd_
            )

            loss = - 1 / m * (np.dot(y.T, np.log(p)) + np.dot((1 - y.T), np.log(1 - p)))
            loss_val = - 1 / m_val * (np.dot(y_val.T, np.log(p_val)) + np.dot((1 - y_val.T), np.log(1 - p_val)))

            reg = self.lmd / (2 * m) * (np.dot(self.theta[1:].T, self.theta[1:]))

            t_h[step, :] = self.theta.T
            c_h[step] = loss + reg
            c_h_v[step] = loss_val + reg

        return c_h, c_h_v, t_h

    def predict(self, X, thrs=0.5):
        X_pred = np.c_[np.ones(X.shape[0]), X]
        return self._sigmoid(np.dot(X_pred, self.theta)) >= thrs

    def compute_performance(self, X, y):
        p = self.predict(X)

        cm = self.compute_matrix(y, p)

        accuracy = self._accuracy(cm)
        precision = self._precision(cm)
        recall = self._recall(cm)
        specificity = self._specificity(cm)
        error_rate = self._error_rate(cm)
        f_measure = self._f_measure(cm)
        false_positive_rate = self._false_positive_rate(cm)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'error_rate': error_rate,
            'f_measure': f_measure,
            'false_positive_rate': false_positive_rate,
        }

    def compute_matrix(self, y, pred):
        tp = tn = fp = fn = 0

        for a, p in zip(y, pred):
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

        return np.array([[tn, fp], [fn, tp]])

    def _accuracy(self, cm):
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        return (tp + tn) / (tn + fp + fn + tp)

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
        return tn / (tn + fp)

    def _error_rate(self, cm):
        return 1 - self._accuracy(cm)

    def _false_positive_rate(self, cm):
        return 1 - self._specificity(cm)

    def _f_measure(self, cm):
        precision = self._precision(cm)
        recall = self._recall(cm)
        return 2 * precision * recall / (precision + recall)
