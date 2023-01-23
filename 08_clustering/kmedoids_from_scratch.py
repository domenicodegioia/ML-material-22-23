from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


class KMedoids(object):
    def __init__(self, n_clusters=2, dist=euclidean_distances, random_state=42):
        self.K = n_clusters
        self.dist = dist
        self.rstate = np.random.RandomState(random_state)
        self.medoids = []

    def fit(self, X):
        # Randomly initialize K cluster centroids
        indices = [self.rstate.randint(X.shape[0])]  # indices of initial medoids
        for _ in range(self.K - 1):
            i = self.rstate.randint(X.shape[0])
            # check if point i is already extracted
            while i in indices:
                i = self.rstate.randint(X.shape[0])
            indices.append(i)
        # at this point: len(indices) = K
        self.medoids = X[indices, :]

        cost, y_pred = self.compute_cost(X, indices)

        new_cost = cost
        new_y_pred = y_pred.copy()

        # only to start the first loop because in the first loop: new_cost = cost
        initial = True
        while (new_cost < cost) | initial:
            initial = False

            cost = new_cost
            y_pred = new_y_pred

            for k in range(self.K):
                # for each prediction belonging to a specific cluster
                for r in [i for i, x in enumerate(new_y_pred == k) if x]:
                    # if the point is not a centroid
                    if r not in indices:
                        indices_temp = indices.copy()
                        indices_temp[k] = r

                        new_cost_temp, y_pred_temp = self.compute_cost(X, indices_temp)

                        if new_cost_temp < new_cost:
                            new_cost = new_cost_temp
                            new_y_pred = y_pred_temp
                            indices = indices_temp

        self.medoids = X[indices, :]

    def compute_cost(self, X, indices):
        # returns cost and labels
        y_pred = np.argmin(self.dist(X, X[indices,:]), axis=1)
        cost = np.sum([np.sum(self.dist(X[y_pred == i], X[[indices[i]], :])) for i in set(y_pred)])
        return cost, y_pred

    def predict(self, X):
        # measure the distance between each new sample and K medoids
        # and assign the nearest medoid to each point
        return np.argmin(self.dist(X, self.medoids), axis=1)
