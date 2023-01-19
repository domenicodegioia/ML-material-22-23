import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

SEED = 42

bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
# bc.data -> n = 30, m = 569
# bc.data non contiene la target variable (diagnosis)
df['diagnosis'] = bc.target

X = df.iloc[::-1].values
y = df['diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

parameters = {
    'penalty': ['l1', 'l2'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'eta0': [0.0001, 0.001, 0.01, 0.1]
}

log_reg = SGDClassifier(early_stopping=False,  # to terminate training when validation score is not improving
                        learning_rate='constant',
                        random_state=SEED)

clf = GridSearchCV(estimator=log_reg,
                   param_grid=parameters,
                   scoring='accuracy',  # Strategy to evaluate the performance of the CV model on the test set
                   cv=5,  # to specify the number of folds
                   verbose=3)  # dettagli richiesti (0 per non printare nulla)

clf.fit(X_train, y_train)

print("Tuned Hyperparameters:", clf.best_params_)
print("Score:", clf.best_score_)

newl = SGDClassifier(alpha=1, learning_rate='constant', penalty='l2', eta0=0.01)
newl.fit(X_train, y_train)

y_pred = newl.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy :", accuracy_score(y_test, y_pred))

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

plot_confusion_matrix(newl, X_test, y_test, cmap='Blues')
plt.show()

from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
