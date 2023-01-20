import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


df = pd.read_csv('../data/glass.csv')
features = df.columns[:-1].tolist()

X = df[features]
y = df['Type']

test_size = 0.2
seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

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

grid = GridSearchCV(estimator=pipeline, param_grid=params, cv=kfold,
                    scoring='accuracy', verbose=1, n_jobs=-1)

grid.fit(X_train, y_train)

print('Best score: ', grid.best_score_)
print('Best params: ', grid.best_params_)

plt.figure(figsize=(9, 6))

train_sizes, train_scores, test_scores = learning_curve(estimator=grid.best_estimator_,
                                                        X=X_train,
                                                        y=y_train,
                                                        train_sizes=np.arange(0.1, 1.1, 0.1),
                                                        cv=kfold,
                                                        scoring='accuracy',
                                                        n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, label='train score', color='blue', marker='o')
plt.fill_between(train_sizes, train_scores_mean + train_scores_std,
                 train_scores_mean - train_scores_std, color='blue', alpha=0.1)
plt.plot(train_sizes, test_scores_mean, label='test score', color='red', marker='o')
plt.fill_between(train_sizes, test_scores_mean + test_scores_std,
                 test_scores_mean - test_scores_std, color='red', alpha=0.1)
plt.title('Learning curve for RFC')
plt.xlabel('Number of training points')
plt.ylabel('Accuracy')
plt.grid(ls='--')
plt.legend(loc='best')
plt.show()
