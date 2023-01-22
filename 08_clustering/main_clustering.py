import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import silhouette_score

import kmeans_from_scratch
import kmedoids_from_scratch

SEED = 88

X, y = datasets.make_blobs(n_samples=500,
                           n_features=2,
                           centers=4,
                           cluster_std=1,
                           center_box=(-10.0, 10.0),
                           shuffle=True,
                           random_state=SEED)

K = 10

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# K-Means from scratch
kmeans_model = kmeans_from_scratch.KMeans(n_clusters=K)
kmeans_model.fit(X)
preds = kmeans_model.predict(X)
print(f"Silhouette Coefficient K-Means (SK):\t{silhouette_score(X, preds)}")
# print(f"Silhouette Coefficient K-Means (FROM SCRATCH):\t{kmeans_model.silhouette_coeff(X, preds)}")

plt.scatter(X[:, 0], X[:, 1], c=preds, s=4)
plt.title("K-Means, K=%i" % K)
plt.show()

# K-Medoids from scratch
kmedoids_model = kmedoids_from_scratch.KMedoids(n_clusters=K, random_state=46)
kmedoids_model.fit(X)
preds = kmedoids_model.predict(X)
print(f"Silhouette Coefficient Home K-Medoids:\t{silhouette_score(X, preds)}")

plt.scatter(X[:, 0], X[:, 1], c=preds, s=4)
plt.title("K-Medoids, K=%i" % K)
plt.show()
