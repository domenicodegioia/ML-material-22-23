import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE


df = pd.read_csv('../data/cardio.csv')

training = df[df.y == 0].drop(columns='y').values
test = df[df.y == 1].drop(columns='y').values

plt.scatter(training[:, 1], training[:, 2])
plt.show()

X = df.drop(columns='y').values
y = df['y'].values

# t-SNE [1] is a tool to visualize high-dimensional data. It converts similarities between data points to joint
# probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the
# low-dimensional embedding and the high-dimensional data.
tsne = TSNE(n_components=2)
# Fit X into an embedded space and return that transformed output.
data_viz = tsne.fit_transform(X)
plt.scatter(data_viz[:, 0], data_viz[:, 1], c=y)
plt.show()

# Fit parameters of GMM via EM algorithm
gmm = GaussianMixture(n_components=10, covariance_type='full')
gmm.fit(training)

print(gmm.weights_.dot(gmm.predict_proba(training).T))
predictions = gmm.weights_.dot(gmm.predict_proba(test).T)

eps = 0.3
print(sum(predictions < eps) / len(predictions))
