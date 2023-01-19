import numpy as np
import pandas as pd
from PCA import pca
from sklearn.metrics import silhouette_score
from kmeans import Kmeans1
from kmedoids import Kmedoids
from sklearn.cluster import KMeans

data = pd.read_csv('./mnist.csv')

x = data.loc[:, data.columns != 'label'].values
y = data.loc[:, data.columns == 'label'].values


'''
# z score
mean = x.mean(axis=0)
dev = x.std(axis=0)
to_del = []
print(x.shape)
for i in range(x.shape[1]):
    if dev[i] != 0:
        x[:, i] = (x[:, i] - mean[i]) / dev[i]
    else:
        to_del.append(i)
# x = np.delete(x, to_del, axis=1)

'''


# data reduction with pca
for i in range(400, x.shape[1], 100):
    print(i)
    x_reduced, var = pca(x, red_index=i)
    if var >= 0.95:
        break
print(x_reduced.shape, var)



# CLUSTERING
k = 10

# KMEANS NORMAL

kobj = KMeans(n_clusters=10, random_state=42)
kobj.fit(x_reduced)
pred = kobj.predict(x_reduced)
sil = silhouette_score(x_reduced, pred)
print('Score kmeans library: {}'.format(sil))

# K-MEANS
km = Kmeans1(ncluster=k, randomstate=42)
km.fit(x_reduced)
pred = km.predict(x_reduced)
sil = silhouette_score(x_reduced, pred)
print('Score Kmeans: {} \n'.format(sil))



# K-MEDOIDS
kmed = Kmedoids(ncluster=k)
kmed.fit(x_reduced)
pred = kmed.predict(x_reduced)
sil = silhouette_score(x_reduced, pred)
print('Score Kmedoids: {}'.format(sil))






