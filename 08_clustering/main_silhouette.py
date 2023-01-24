from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score
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

K = range(2, 10)

silhouettes = []
map_silhouette = {}

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=SEED)
    kmeans.fit(X)
    preds = kmeans.predict(X)

    sil = silhouette_score(X, preds)
    silhouettes.append(sil)
    map_silhouette[k] = sil

for k, s in map_silhouette.items():
    print(f"{k}:\t{s}")

plt.plot(K, silhouettes, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette coefficient')
plt.title('Silhouette Analysis')
plt.show()
