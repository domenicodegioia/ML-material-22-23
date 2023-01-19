
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn import mixture, datasets
import hdbscan
from sources.metrics import silhouette_score
from sources.kmedoids2 import Kmedoids

# import matplotlib.pyplot as plt
from sources.utils import get_zscore, apply_zscore
from sources.kmeans import KMeans as HKMeans
from sources.kmedoids import KMedoids

# heart = pd.read_csv('../data/heart.csv')
# proc_data = heart.loc[:, ["age","trestbps"]].values

proc_data, y_true = datasets.make_blobs(n_samples=500,
                                        n_features=2,
                                        centers=4,
                                        cluster_std=1,
                                        center_box=(-10.0, 10.0),
                                        shuffle=True,
                                        random_state=88)  # For reproducibility
#
# proc_data, y_true = datasets.make_circles(n_samples=500,
#                                           factor=.5,
#                                           noise=.05,
#                                           random_state=88)

# proc_data, y_true = datasets.make_moons(n_samples=500,
#                                         noise=.05,
#                                         random_state=88)

k_s = 4

means, stds = get_zscore(proc_data)
proc_data = apply_zscore(proc_data, means, stds)
'''plt.figure(figsize=(10.20, 5.76))
plt.subplot(241)
plt.scatter(proc_data[:, 0], proc_data[:, 1], s=4)
plt.title("Original data")'''

print('PROC DATA:', proc_data.shape)

#################################################################################################
# K-Means
kmeans_obj = KMeans(n_clusters=k_s, random_state=42)
kmeans_obj.fit(proc_data)
y_pred = kmeans_obj.predict(proc_data)
print(f"Silhouette Coefficient K-Means:\t{silhouette_score(proc_data, y_pred)}")

'''plt.subplot(242)
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred, s=4)
plt.title("K-Means")'''

# K-Means from scratch
kmeans_obj_2 = HKMeans(n_clusters=k_s, random_state=42)
kmeans_obj_2.fit(proc_data)
y_pred = kmeans_obj_2.predict(proc_data)
# print(f"Silhouette Coefficient Home K-Means:\t{silhouette_score(proc_data,y_pred)}")

'''plt.subplot(243)
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred, s=4)
plt.title("K-Means")'''


#################################################################################################
# K-Medoids from scratch
kmedoids_obj = KMedoids(n_clusters=k_s, random_state=46)
kmedoids_obj.fit(proc_data)
y_pred = kmedoids_obj.predict(proc_data)
print(f"Silhouette Coefficient Home K-Medoids:\t{silhouette_score(proc_data,y_pred)}")

'''plt.subplot(244)
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred, s=4)
plt.title("K-Medoids")
'''
#################################################################################################
# KMEDOIDS2MIO
kmed = Kmedoids(ncluster=k_s, randomstate=46)
kmed.fit(proc_data)
pred = kmed.predict(proc_data)
print(f"Silhouette Coefficient Home K-Medoids MIO:\t{silhouette_score(proc_data,pred)}")

#################################################################################################
# Gaussian Mixture Model
cv_types = ['spherical', 'tied', 'diag', 'full']
gmm = mixture.GaussianMixture(n_components=k_s, covariance_type='full', random_state=42)
gmm.fit(proc_data)
y_pred = gmm.predict(proc_data)
gmm.bic(proc_data)
print(f"Silhouette Coefficient Gaussian Mixture Model:\t{silhouette_score(proc_data,y_pred)}")

plt.subplot(245)
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred, s=4)
plt.title("Gaussian Mixture Model")

#################################################################################################
# DBSCAN
db = DBSCAN(eps=0.3, min_samples=20)
# db = DBSCAN(eps=0.3, min_samples=5)
y_pred = db.fit_predict(proc_data)
print(f"Silhouette Coefficient DBSCAN:\t{silhouette_score(proc_data[y_pred != -1],y_pred[y_pred != -1])}")

plt.subplot(246)
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred, s=4)
plt.title("DBSCAN")

#################################################################################################
# HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=20)
# hdb = hdbscan.HDBSCAN(min_cluster_size=5)
y_pred = hdb.fit_predict(proc_data)
print(f"Silhouette Coefficient HDBSCAN:\t{silhouette_score(proc_data[y_pred != -1],y_pred[y_pred != -1])}")

plt.subplot(247)
plt.scatter(proc_data[:, 0], proc_data[:, 1], c=y_pred, s=4)
plt.title("HDBSCAN")

#################################################################################################


plt.show()