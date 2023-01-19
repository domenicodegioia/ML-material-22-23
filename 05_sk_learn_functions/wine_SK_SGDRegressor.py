import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve

plt.style.use(['ggplot'])

wine_1 = pd.read_csv('../data/wine.csv', header=None)
wine_1 = wine_1.dropna(axis=1)
# wine = wine_1.sample(frac=1).reset_index(drop=True)

X = wine_1.iloc[:, :-1].values
y = wine_1.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

sgdr = SGDRegressor(random_state=42)
sgdr.fit(X_train, y_train)

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=sgdr,
    X=X_train,
    y=y_train,
    random_state=42)

train_scores_mean = -train_scores.mean(axis=1)
validation_scores_mean = -validation_scores.mean(axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, validation_scores_mean, label='Validation error')
plt.ylabel('MSE', fontsize=14)
plt.xlabel('Training set size', fontsize=14)
plt.title('Learning curves for a linear regression model', fontsize=18, y=1.03)
plt.legend()
# plt.ylim(0, 40)
plt.show()
