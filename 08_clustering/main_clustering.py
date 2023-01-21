import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import silhouette_score

from kmeans_from_scratch import KMeans
from kmedoids_from_scratch import KMedoids

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
kmeans_model = KMeans(n_clusters=K)
kmeans_model.fit(X)
preds = kmeans_model.predict(X)
print(f"Silhouette Coefficient Home K-Means:\t{silhouette_score(X, preds)}")

plt.scatter(X[:, 0], X[:, 1], c=preds, s=4)
plt.title("K-Means, K=%i" % K)
plt.show()

# K-Medoids from scratch
kmedoids_model = KMedoids(n_clusters=K, random_state=46)
kmedoids_model.fit(X)
preds = kmedoids_model.predict(X)
print(f"Silhouette Coefficient Home K-Medoids:\t{silhouette_score(X, preds)}")

plt.scatter(X[:, 0], X[:, 1], c=preds, s=4)
plt.title("K-Medoids, K=%i" % K)
plt.show()
