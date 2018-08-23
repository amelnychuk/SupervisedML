
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import load_digits

import warnings

def show_digit(digit, label, num):
    plt.gray()
    plt.matshow(digit[num].reshape((8,8)))
    plt.title("Training: {}".format(label[num]))
    plt.show()

def get_data(limit=None):
    """Gets mnist data from lazyprogrammers' hit hub.

    :param limit:
        (int) the maximum amount of samples your would like from mnist

    :return: np.ndarray (N,D), np.ndarray (N,)
        X, Y mnist features and labels
    """
    data, labels = load_digits(return_X_y=True)

    X, Y = shuffle(data, labels)
    if limit is not None:
        xSize = len(X)
        if limit > xSize:
            warnings.warn("Limit of {} exceeds data sample size of {}".format(limit, xSize))
        X, Y = X[:limit], Y[:limit]

    return X, Y

