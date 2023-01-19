from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv('../data/glass.csv')

# distribution of target class
print(df['Type'].value_counts())

# SKEWNESS FOR EACH FEATURE
# We want to plot the Skewness value: ideally we want this value close to 0.
# Skewness is a measure of asymmetry or distortion of symmetric distribution.
# It measures the deviation of the given distribution of a random variable from a symmetric distribution,
# such as normal distribution. A normal distribution is without any skewness, as it is symmetrical on
# both sides. Hence, a curve is regarded as skewed if it is shifted towards the right or the left.
features = df.columns[:-1].tolist()  # name of column in dataframe
for f in features:
    skew = df[f].skew()
    sns.displot(df[f], kde=False, label='Skew = %.3f' % (skew), bins=30)
    plt.legend(loc='best')
    plt.show()


# RIMOZIONE OUTLIERS
def outlier_hunt(df):
    outlier_indices = []

    for col in df.columns.tolist():
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IRQ = Q3 - Q1
        outlier_step = 1.5 * IRQ

        outlier_list_col = df[(df[col] < Q1 - outlier_step) |
                              (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    # Counter, ricevendo una lista, restituisce un dizionario avente come
    # keys i valori possibili della lista (distinti) e come values
    # il relativo numero di occorrenze di quel valore nella lista
    outlier_indices = Counter(outlier_indices)

    # lista dei samples che contengono outliers per almeno 3 feature
    # da eliminare successivamente
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > 2)

    return multiple_outliers


outlier_indices = outlier_hunt(df[features])
print(f'Dataset has {len(outlier_indices)} outliers')
df = df.drop(outlier_indices, axis=0).reset_index(drop=True)

# boxplot solution (display distribution with outliers)
plt.figure(figsize=(8, 6))
sns.boxplot(df[features])  # set showfliers=False to remove the outliers from the chart
plt.show()


# HEATMAP
# correlation matrix between all the features we are examining and our y-variable
corr = df[features].corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2',
            annot_kws={'size': 15}, xticklabels=features,
            yticklabels=features, alpha=0.7, cmap='coolwarm')
plt.title('Correlation Heatmap', fontdict={'fontsize': 38}, pad=32)
plt.show()

# SPLITTING -> SCALING -> DIMENSIONALITY REDUCTION
# Pay attention to the order! It is important to perform normalization first and then
# feature selection based on the components identified with PCA.

# SPLITTING
SEED = 42
X = df[features]
y = df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# SCALING
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# DIMENSIONALITY REDUCTION
# We identify how many components are needed to contain a variance above a threshold of about 80%.
# For this purpose we use a comulative sum to identify the optimal value.
pca = PCA(random_state=SEED)
pca.fit(X_train)
var_exp = pca.explained_variance_ratio_
cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(cum_var_exp) + 1), var_exp, align='center', label='individual variance explained', alpha=0.7)
plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='cumulative variance explained', color='red')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.xticks(np.arange(1, len(var_exp) + 1, 1))
plt.legend(loc='center right')
plt.show()
for i, _ in enumerate(cum_var_exp):
    print("PC" + str(i + 1), f"Cumulative variance: {cum_var_exp[i] * 100} %")

pca = PCA(n_components=5, random_state=SEED)
X_train = pca.fit_transform(X_train, X_test)
X_test = pca.transform(X_test)

# SVC + KFold + GridSearch

kfold = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
params = {
    'C': [10, 50, 100],
    'kernel': ('linear', 'rbf', 'sigmoid'),
    'tol': [1e-2, 1e-3, 1e-4]
}

grid = GridSearchCV(estimator=SVC(), param_grid=params, cv=kfold,
                    scoring='accuracy', verbose=1, n_jobs=1)

grid.fit(X_train, y_train)

print('Best score: ', grid.best_score_)
print('Best params:', grid.best_params_)


# build the learning curves for the best model

plt.figure(figsize=(9, 6))
# train_sizes: Numbers of training examples
# train_scores: Scores on training sets.
# test_scores: Scores on test set.
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

# build the confusion matrix for the best model
np.set_printoptions(precision=2)

class_names = np.unique(y)

disp = ConfusionMatrixDisplay.from_estimator(estimator=grid.best_estimator_,
                                             X=X_test,
                                             y=y_test,
                                             display_labels=class_names,
                                             cmap=plt.cm.Blues,
                                             normalize=None)
disp.ax_.set_title("Confusion matrix, without normalization")
plt.show()

disp = ConfusionMatrixDisplay.from_estimator(estimator=grid.best_estimator_,
                                             X=X_test,
                                             y=y_test,
                                             display_labels=class_names,
                                             cmap=plt.cm.Blues,
                                             normalize='true')
disp.ax_.set_title("Normalized confusion matrix")
plt.show()
