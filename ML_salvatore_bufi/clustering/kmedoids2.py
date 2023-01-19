import numpy as np
from sklearn.metrics import euclidean_distances

class Kmedoids(object):
    def __init__(self, ncluster=10, dist=euclidean_distances, randomstate=42):
        self.ncluster = ncluster
        self.dist = dist
        self.rstate = np.random.RandomState(randomstate)
        self.center = []


    def cost(self, x, indices):
        y = np.argmin(self.dist(x, x[indices, :]), axis=1)

        cost = np.sum([np.sum(self.dist(x[y == i, :], x[[indices[i]], :])) for i in set(y)])
        return y, cost

    def fit(self, x):
        indices = [self.rstate.randint(x.shape[0])]
        for _ in range(self.ncluster - 1):
            index = self.rstate.randint(x.shape[0])
            if index in indices:
                index = self.rstate.randint(x.shape[0])
            indices.append(index)
        self.center = x[indices, :]

        y, cost = self.cost(x, indices)
        initial = True
        cost_new = cost
        y_new = y

        while (cost_new < cost) | initial:
            initial = False
            y = y_new
            cost = cost_new

            for k in range(self.ncluster):
                for r in [i for i, x in enumerate(y) if x]:
                    if r not in indices:
                        indices_temp = indices.copy()
                        indices_temp[k] = r
                        y_temp, cost_temp = self.cost(x, indices_temp)
                        if cost_temp < cost:
                            indices = indices_temp
                            y_new = y_temp
                            cost_new = cost_temp
        self.center = x[indices, :]


    def predict(self, x):
        return np.argmin(self.dist(x, self.center), axis=1)
