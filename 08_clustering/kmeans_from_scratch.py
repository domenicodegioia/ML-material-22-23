from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KMeans(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        self.K = n_clusters
        self.dist = dist
        self.rstate = np.random.RandomState(random_state)
        self.centroids = []

    def fit(self, X):
        # Randomly initialize K cluster centroids
        rint = self.rstate.randint
        initial_indices = [rint(X.shape[0])]  # indices of initial centroids
        for _ in range(self.K - 1):
            i = rint(X.shape[0])
            # check if point i is already extracted
            while i in initial_indices:
                i = rint(X.shape[0])
            initial_indices.append(i)
        # at this point: len(initial_indices) = K
        self.centroids = X[initial_indices, :]

        while True:
            old_centroids = self.centroids.copy()

            self.y_pred = self.predict(X)

            # recalculate means (centroids) for obervations assigned to each cluster
            for i in set(self.y_pred):
                self.centroids[i] = np.mean(X[self.y_pred == i], axis=0)

            # stop when the clustering did not change at all during the last iteration
            if (old_centroids == self.centroids).all():
                break


    def predict(self, X):
        # measure the distance between each new sample and K centroids
        # and assign the nearest centroid to each point
        return np.argmin(self.dist(X, self.centroids), axis=1)
