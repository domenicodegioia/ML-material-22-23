# import required modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate a random regression problem.

# n_informative: The number of informative features, i.e., the number of features used to build the linear model used to generate the output.
# noise: The standard deviation of the gaussian noise applied to the output.
# random_state:
# If int, random_state is the seed used by the random number generator;
# If RandomState instance, random_state is the random number generator;
# If None, the random number generator is the RandomState instance used by np.random.
X, y = make_regression(n_samples=100,
                       n_features=1,
                       n_informative=1,
                       noise=10,
                       random_state=10)

# Number of training examples
m = X.shape[0]

# Plot the generated data set.
plt.scatter(X, y, s=30, marker='o')  # s: marker size
plt.xlabel("Feature_1 --->")
plt.ylabel("Target_Variable --->")
plt.title('Simple Linear Regression')
plt.show()

# Convert target variable array from 1d to 2d with 1 column.
y = y.reshape(100, 1)
# In general: y = y.reshape(m,-1)

# Adding x0=1 to each instance of x (adding the first column with values 1)
X_new = np.array([np.ones(m), X.flatten()]).T

# Normal Equation --> theta_hat = (X^T * X)^-1 * X^T * y
theta_best_values_batch_gd = np.linalg.inv(X_new.T.dot(X_new)).dot(X_new.T).dot(y)
print('Best value of theta with Normal_Equations_Batch_GD\n', theta_best_values_batch_gd)

# sample data instance for the prediction
X_sample = np.array([[-2], [4]])
m_new = len(X_sample)

# Adding x0=1 to each instance of x_sample (adding the first column with values 1)
X_sample_new = np.array([np.ones(m_new), X_sample.flatten()]).T

# predicted values for given data instance --> y^(i) = theta^T * x^(i)
predict_value = X_sample_new.dot(theta_best_values_batch_gd)
print('Predicted value with Normal Equations\n', predict_value)

# Plot the output.
plt.scatter(X, y, s=30, marker='o')  # s: marker size
plt.plot(X_sample, predict_value, c='red')  # c: color
plt.plot()
plt.xlabel("Feature_1 --->")
plt.ylabel("Target_Variable --->")
plt.title('Simple Linear Regression')
plt.show()

# Verification of the model prevision
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, y)
print("Best value of theta from Sklearn:", lr.intercept_, lr.coef_, sep='\n')
print("Predicted value from Sklearn :", lr.predict(X_sample), sep='\n')
