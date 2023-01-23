import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


SEED = 42

X, y = datasets.make_blobs(n_samples=500,
                           n_features=2,
                           centers=4,
                           cluster_std=1,
                           center_box=(-10.0, 10.0),
                           shuffle=True,
                           random_state=SEED)

K = 10

sc = StandardScaler()
X = sc.fit_transform(X)

#################################################################################################
# KMeans
kmeans = KMeans(n_clusters=K, random_state=SEED, n_init='auto')
# n_init: Number of times the k-means algorithm is run with different centroid seeds.
# The final results is the best output of n_init consecutive runs in terms of inertia.
kmeans.fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
preds = kmeans.predict(X)
print(f"Silhouette Coefficient K-Means:\t{silhouette_score(X, preds)}")
plt.scatter(X[:, 0], X[:, 1], c=preds, s=4)
plt.title("K-Means, K=%i" % K)
plt.show()

#################################################################################################
# Gaussian Mixture Model
gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=42)
# cv_types = ['spherical', 'tied', 'diag', 'full']
gmm.fit(X)
print(gmm.means_)
preds = gmm.predict(X)
print(f"Bayesian information criterion (BIC) GMM:\t{gmm.bic(X)}")
print(f"Silhouette Coefficient GMM:\t{silhouette_score(X, preds)}")
plt.scatter(X[:, 0], X[:, 1], c=preds, s=4)
plt.title("GMM, K=%i" % K)
plt.show()

#################################################################################################
# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=20)
preds = dbscan.fit_predict(X)
print(f"Silhouette Coefficient DBSCAN:\t{silhouette_score(X, preds)}")
plt.scatter(X[:, 0], X[:, 1], c=preds, s=4)
plt.title("DBSCAN, K=%i" % K)
plt.show()
