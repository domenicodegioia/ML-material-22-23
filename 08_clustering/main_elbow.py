from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

SEED = 42

# Creating and Visualizing the data
X, y = datasets.make_blobs(n_samples=1000,
                           n_features=2,
                           centers=4,
                           cluster_std=2,
                           center_box=(-10.0, 10.0),
                           shuffle=True,
                           random_state=SEED)

sc = StandardScaler()
X = sc.fit_transform(X)

plt.scatter(X[:, 0], X[:, 1], marker="o", c=y, s=15, edgecolor="k")
plt.title("Dataset")
plt.show()

# Building the clustering model and calculating the values of the Distortion and Inertia for each value of k
distortions = []
inertias = []
map_distortions = {}
map_intertias = {}
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=SEED)
    kmeans.fit(X)

    # Distortion: average of the squared distances from the cluster centers of the respective clusters
    dist = np.mean(np.min(euclidean_distances(X, kmeans.cluster_centers_), axis=1))
    distortions.append(dist)
    map_distortions[k] = dist

    #Inertia: sum of squared distances of samples to their closest cluster center
    iner = kmeans.inertia_
    inertias.append(iner)
    map_intertias[k] = iner

# Tabulating and Visualizing the results

# Using the different values of Distortion:
for k, d in map_distortions.items():
    print(f"{k}:\t{d}")

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

# Using the different values of Inertia
for k, i in map_intertias.items():
    print(f"{k}:\t{i}")

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

# To determine the optimal number of clusters, we have to select the value of k at the “elbow”
# ie the point after which the distortion/inertia start decreasing in a linear fashion.
# Thus for the given data, we conclude that the optimal number of clusters for the data is 3.


# Reference: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/