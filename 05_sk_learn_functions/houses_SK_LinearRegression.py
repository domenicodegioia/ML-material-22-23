from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

houses = pd.read_csv('../data/houses.csv')

houses_df = pd.DataFrame(houses)

print(houses_df.columns)


houses['LotFrontage'] = houses['LotFrontage'].fillna(houses['LotFrontage'].mean())
houses = houses.dropna(axis=1)

x = houses[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
            'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces']].values
y = houses['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=20)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print('Score: {}'.format(lr.score(X_test, y_test)))
print('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
print('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))
print('R2: {}'.format(r2_score(y_test, y_pred)))

train_sizes, train_scores, test_scores = learning_curve(estimator=lr, X=X_train, y=y_train,
                                                        cv=5, train_sizes=np.linspace(0.1, 1.0),
                                                        n_jobs=-1)
# Calculate training and test mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
y_pred=lr.predict(X_test)
print(y_pred)

# Plot the learning curve

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
#Fill the area between two horizontal curves
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model accuracy')
plt.grid()
plt.legend(loc='lower right')
plt.show()



