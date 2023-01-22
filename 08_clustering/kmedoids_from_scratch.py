from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KMedoids(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        self.K = n_clusters
        self.dist = dist
        self.rstate = np.random.RandomState(random_state)
        self.medoids = []
        self.indices = []  # indices of medoids

    def fit(self, X):
        # Randomly initialize K cluster centroids
        rint = self.rstate.randint
        self.indices = [rint(X.shape[0])]  # indices of initial medoids
        for _ in range(self.K - 1):
            i = rint(X.shape[0])
            # check if point i is already extracted
            while i in self.indices:
                i = rint(X.shape[0])
            self.indices.append(i)
        # at this point: len(initial_indices) = K
        self.medoids = X[self.indices, :]

        # measure the distance between each point and K medoids
        # and assign to each point the nearest medoid
        self.y_pred = self.predict(X)
        cost, _ = self.compute_cost(X, self.indices)

        new_cost = cost
        new_y_pred = self.y_pred.copy()
        new_indices = self.indices[:]

        # only to start the first loop because in the first loop: new_cost = cost
        initial = True
        while (new_cost < cost) | initial:
            initial = False
            cost = new_cost
            self.y_pred = new_y_pred
            self.indices = new_indices
            for k in range(self.K):
                for r in [i for i, x in enumerate(new_y_pred == k) if x]:
                    if r not in self.indices:
                        indices_temp = self.indices[:]
                        indices_temp[k] = r
                        new_cost_temp, y_pred_temp = self.compute_cost(X, indices_temp)
                        if new_cost_temp < new_cost:
                            new_cost = new_cost_temp
                            new_y_pred = y_pred_temp
                            new_indices = indices_temp

        self.medoids = X[self.indices, :]

    def compute_cost(self, X, indices):
        # returns cost and labels
        y_pred = np.argmin(self.dist(X, X[indices,:]), axis=1)
        return np.sum([np.sum(self.dist(X[y_pred == i], X[[indices[i]], :])) for i in set(y_pred)]), y_pred

    def predict(self, X):
        # measure the distance between each new sample and K medoids
        # and assign the nearest medoid to each point
        return np.argmin(self.dist(X, self.medoids), axis=1)
