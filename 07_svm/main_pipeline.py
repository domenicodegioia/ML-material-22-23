import numpy as np  # linear algebra
import pandas as pd  # read and wrangle dataframes
import matplotlib.pyplot as plt  # visualization
import seaborn as sns  # statistical visualizations and aesthetics
from sklearn.preprocessing import (FunctionTransformer, StandardScaler)  # preprocessing
from sklearn.decomposition import PCA  # dimensionality reduction
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV,
                                     learning_curve)  # model selection modules
from sklearn.pipeline import Pipeline  # streaming pipelines

import warnings
from sklearn.svm import SVC
from utils import plot_skew, plot_learning_curve, outlier_hunt

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

df = pd.read_csv('../../../../../Downloads/FML_6/data/glass.csv')
features = df.columns[:-1].tolist()
print(df.shape)

print(df.head(5))

print(df['Type'].value_counts())

# We want to plot the Skewness value: ideally we want this value close to 0.
# Skewness is a measure of asymmetry or distortion of symmetric distribution.
# It measures the deviation of the given distribution of a random variable from a symmetric distribution,
# such as normal distribution. A normal distribution is without any skewness, as it is symmetrical on
# both sides. Hence, a curve is regarded as skewed if it is shifted towards the right or the left.
plot_skew(df[features])


print(f'Dataset has {len(outlier_hunt(df[features]))} elements as outliers')

plt.figure(figsize=(8,6))
sns.boxplot(df[features])
plt.show()

corr = df[features].corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2',
            annot_kws={'size': 15}, xticklabels=features,
            yticklabels=features, alpha=0.7, cmap='coolwarm')
plt.show()

outlier_indices = outlier_hunt(df[features])
df = df.drop(outlier_indices).reset_index(drop=True)
print(df.shape)

plot_skew(df[features])

print(df['Type'].value_counts())

X = df[features]
y = df['Type']

test_size = 0.2
seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# Pay attention to the order! It is important to perform normalization first and then feature
# selection based on the components identified with PCA.
# In this case we use SciKit-Learn's Pipeline to perform these operations sequentially.
pipeline = Pipeline([
                      ('sc', StandardScaler()),
                      ('pca', PCA(n_components=5, random_state=seed)),
                      ('SVC', SVC(random_state=seed))])

kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
params = [{
    'SVC__C': [10, 50, 100],
    'SVC__kernel': ('linear', 'rbf', 'sigmoid'),
    'SVC__tol': [1e-2, 1e-3, 1e-4]
}]
grid = GridSearchCV(pipeline, param_grid=params, cv=kfold,
             scoring='accuracy', verbose=1, n_jobs=-1)

grid.fit(X_train, y_train)

print('--------BEST SCORE----------')
print(grid.best_score_*100)
print('--------BEST PARAM----------')
print(grid.best_params_)

plt.figure(figsize=(9,6))

train_sizes, train_scores, test_scores = learning_curve(
              estimator= grid.best_estimator_ , X= X_train, y = y_train,
                train_sizes=np.arange(0.1,1.1,0.1), cv= 10,  scoring='accuracy', n_jobs= - 1)

plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for RFC')