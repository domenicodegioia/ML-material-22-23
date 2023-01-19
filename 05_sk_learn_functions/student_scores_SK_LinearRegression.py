import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_learning_curves

SEED = 42

df = pd.read_csv('../data/student_scores.csv')

print(df.head())
print(df.shape)

df.plot.scatter(x='Hours', y='Scores', title='Scatterplot of hours and scores percentages')
plt.show()

print(df.corr())

X = df['Hours'].values.reshape(-1, 1)
y = df['Scores'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print('Intercept:', regressor.intercept_)
print('Slope:', regressor.coef_)


def calc(slope, intercept, hours):
    return slope * hours + intercept


x_sample = 9.5
pred = calc(regressor.coef_, regressor.intercept_, x_sample)
print('sample value:', x_sample)
print('predicted value with calc():', pred)
pred = regressor.predict([[9.5]])
print('predicted value with sklearn function', pred)

y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# print the metrics results using the f string and the 2 digit precision after the comma with :.2f:
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

plot_learning_curves(X_train, y_train, X_test, y_test, regressor)
plt.show()

# Plot outputs
plt.scatter(X_train, y_train, color='b')
plt.plot(X_test, y_pred, color='k')
plt.show()
