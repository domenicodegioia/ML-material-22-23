import numpy as np
from sklearn.metrics import euclidean_distances

class Kmeans1(object):
    def __init__(self, ncluster=10, dist=euclidean_distances, randomstate=42):
        self.ncluster = ncluster
        self.dist = dist
        self.rstate = np.random.RandomState(randomstate)
        self.ypred = []
        self.cluster_center = []

    def fit(self, x):
        index_list = [self.rstate.randint(x.shape[0])]
        for _ in range(self.ncluster - 1):
            index = self.rstate.randint(x.shape[0])
            while index in index_list:
                index = self.rstate.randint(x.shape[0])
            index_list.append(index)
        self.cluster_center = x[index_list, :]

        while True:
            old_center = self.cluster_center.copy()
            self.ypred = np.argmin(self.dist(x, self.cluster_center), axis=1)

            for i in range(self.ncluster):
                self.cluster_center[i] = np.mean(x[self.ypred == i], axis=0)
            if (self.cluster_center == old_center).all():
                break

    def predict(self, x):
        ypred = np.argmin(self.dist(x, self.cluster_center), axis=1)
        return ypred





