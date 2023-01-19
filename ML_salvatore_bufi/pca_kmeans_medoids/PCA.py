import numpy as np

def pca(x, red_index=1):
    m = len(x)
    sigma = ( 1/ m) * (x.T @ x)
    u, s, v = np.linalg.svd(sigma)

    u_reduced = u[:, :red_index]
    x_reduced = x @ u_reduced

    reconstruct = x_reduced @ u_reduced.T

    var = (sum(s[:red_index])) / sum(s)

    return x_reduced, var

