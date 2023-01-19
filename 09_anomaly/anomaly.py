import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE


df = pd.read_csv('../data/cardio.csv')
print(df.head())
print(df.tail())

training = df[df.y == 0].drop(columns='y').values
test = df[df.y == 1].drop(columns='y').values

plt.scatter(training[:, 1], training[:, 2])
plt.show()

data_complete = df.drop(columns='y').values
labels_complete = df['y'].values

tsne = TSNE(n_components=2)
data_viz = tsne.fit_transform(data_complete)
plt.scatter(data_viz[:, 0], data_viz[:, 1], c=labels_complete)
plt.show()

gmm = GaussianMixture(n_components=10, covariance_type='full')
gmm.fit(training)

print(gmm.weights_.dot(gmm.predict_proba(training).T))
predictions = gmm.weights_.dot(gmm.predict_proba(test).T)

eps = 0.3
print(sum(predictions < eps) / len(predictions))
