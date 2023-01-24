import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.U = None

    def fit(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        m = X.shape[0]

        Sigma = (1 / m) * X.T.dot(X)

        U, S, _ = np.linalg.svd(Sigma)

        if self.n_components is None:
            self.U = U
        else:
            self.U = U[:, : self.n_components]

    def transform(self, X):
        return self.U.dot(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)