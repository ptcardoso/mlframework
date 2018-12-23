import numpy as np


def cross_entropy(y, y_pred):
    n = y.shape[0]

    cost = np.multiply(y, np.log(y_pred))
    cost += np.multiply((np.ones(y.shape) - y), np.log(np.ones(y.shape) - y_pred))
    cost *= -1 / n

    return cost.sum()
