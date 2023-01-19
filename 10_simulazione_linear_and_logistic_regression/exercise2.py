import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

SEED = 42

cancer = pd.read_csv("../data/cancer.csv")

# print(cancer.isnull().sum())

cancer = cancer.drop("id", axis=1)

cancer['diagnosis'] = cancer['diagnosis'].replace("B", 0).replace("M", 1)

X = cancer.drop("diagnosis", axis=1).values
y = cancer['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

parameters = {
    'penalty': ['l1', 'l2'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'eta0': [0.0001, 0.001, 0.01, 0.1]
}

log_reg = SGDClassifier(early_stopping=False,
                        learning_rate='constant',
                        random_state=SEED)

clf = GridSearchCV(estimator=log_reg,
                   param_grid=parameters,
                   scoring='accuracy',
                   cv=5,
                   verbose=3)

clf.fit(X_train, y_train)

print("Tuned Hyperparameters:", clf.best_params_)
print("Score:", clf.best_score_)

for p, v in clf.best_params_.items():
    print(p, v)

# model = SGDClassifier(alpha=clf.best_params_['alpha'],
#                       penalty=clf.best_params_['penalty'],
#                       eta0=clf.best_params_['eta0'],
#                       learning_rate='constant')

y_pred = clf.predict(X_test)

print("accuracy: ", accuracy_score(y_test, y_pred))

plot_confusion_matrix(clf, X_test, y_test, cmap='Blues')
plt.show()