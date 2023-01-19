import numpy as np


def get_zscore(data):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return means, stds


def apply_zscore(data, means, stds):
    return (data - means) / stds
