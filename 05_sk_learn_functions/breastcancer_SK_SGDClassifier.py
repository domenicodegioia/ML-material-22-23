from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
SEED=42
bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns = bc.feature_names)
df['diagnosis'] = bc.target
dfX = df.iloc[:,:-1]   # Features - 30 columns
dfy = df['diagnosis']  # Label - last column
X = dfX.values
y = dfy.values
#Standardize features by removing the mean and scaling to unit variance.
sc = StandardScaler()
x_train, x_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=SEED)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
newl=SGDClassifier(alpha= 0.01, learning_rate= 'optimal', penalty= 'l2')
newl.fit(x_train,y_train)
y_pred=newl.predict(x_test)
from sklearn.metrics import accuracy_score
print("Accuracy :",accuracy_score(y_test,y_pred))
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(newl, x_test, y_test, cmap='Blues')
plt.show()
from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
