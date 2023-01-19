import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from linear_regression_homemade import LinearRegression

houses = pd.read_csv('../data/houses.csv')

houses['LotFrontage'] = houses['LotFrontage'].fillna(houses['LotFrontage'].mean())

houses = houses.dropna(axis=1)

houses = houses.sample(frac=1).reset_index(drop=True)

x = houses[['GrLivArea', 'LotArea', 'GarageArea', 'FullBath']].values
y = houses['SalePrice'].values

train_i = round(len(x) * 0.8)

val_i = round(train_i * 0.7)

X_train = x[:train_i]
y_train = y[:train_i]

X_test = x[train_i:]
y_test = y[train_i:]

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = np.c_[np.ones(X_train.shape[0]), X_train]

X_val = X_train[val_i:]
y_val = y_train[val_i:]

X_train = X_train[:val_i]
y_train = y_train[:val_i]

lin_reg = LinearRegression(n_features=X_train.shape[1],
                           n_steps=1000,
                           learning_rate=0.00001,
                           lmd=0.1)

# c_h, c_h_v, t_h = lin_reg.fit_full_batch_gd(X_train, y_train, X_val, y_val)
c_h, c_h_v, t_h = lin_reg.fit_stochastic_gd(X_train, y_train, X_val, y_val)
# c_h, c_h_v, t_h = lin_reg.fit_mini_batch_gd(X_train, y_train, X_val, y_val)

print(f'''Thetas: {*lin_reg.theta,}''')
print(f'''Final train cost/MSE:  {c_h[-1]:.3f}''')
print(f'''Final validation cost/MSE:  {c_h_v[-1]:.3f}''')

# 1) plot loss curves
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_ylabel('J(Theta)')
ax.set_xlabel('Iterations')
c, = ax.plot(range(lin_reg.n_steps), c_h, 'b.')
cv, = ax.plot(range(lin_reg.n_steps), c_h_v, 'r+')
c.set_label('Train cost')
cv.set_label('Valid cost')
ax.legend()
plt.show()

# compute performance of your regressor model
perf = lin_reg.compute_performance(X_test, y_test)
print('\nperformance:\n')
print(''.join(['%s = %s\n' % (key, value) for (key, value) in perf.items()]))

# 2) plot cost history evolution with reference to thetas (it's a 3D plot, please choose only 2 different thetas!!)
_ = plt.figure()

ax = plt.axes(projection='3d')
first_dim = 0
second_dim = 1

# Make grid data.
A = np.linspace(np.min(t_h[:, first_dim]), np.max(t_h[:, first_dim]), 100)
B = np.linspace(np.min(t_h[:, second_dim]), np.max(t_h[:, second_dim]), 100)
A, B = np.meshgrid(A, B)
Z = lin_reg.cost_grid(X_train, y_train, A, B, first_dim, second_dim)

# Plot the surface.
surf = ax.plot_surface(A, B, Z, cmap=cm.Wistia, linewidth=0, antialiased=False)

ax.plot(t_h[:, first_dim], t_h[:, second_dim], c_h, label='parametric curve')
plt.show()

# 3) compute and plot learning curves
c_h, c_h_v = lin_reg.learning_curves(X_train, y_train, X_val, y_val)

_, ax = plt.subplots(figsize=(12, 8))

ax.set_ylabel('J(Theta)')
ax.set_xlabel('# Training samples')
c, = ax.plot(range(len(c_h)), c_h, color="blue")
cv, = ax.plot(range(len(c_h_v)), c_h_v, color="red")
ax.set_yscale('log')
c.set_label('Train cost')
cv.set_label('Valid cost')
ax.legend()

plt.show()
