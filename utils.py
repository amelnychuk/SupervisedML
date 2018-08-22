
import pandas as pd
import csv
import numpy as np
from sklearn.utils import shuffle

import warnings

def get_data(limit=None):
    """Gets mnist data from lazyprogrammers' hit hub.

    :param limit:
        (int) the maximum amount of samples your would like from mnist

    :return: np.ndarray (N,D), np.ndarray (N,)
        X, Y mnist features and labels
    """

    data = pd.read_table("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/mnist_csv/Xtest.txt", sep=',', header=None)
    labels = pd.read_table("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/mnist_csv/label_test.txt", header=None, squeeze=True)

    X, Y = shuffle(data, labels)
    if limit is not None:
        xSize = len(X)
        if limit > xSize:
            warnings.warn("Limit of {} exceeds data sample size of {}".format(limit, xSize))
        X, Y = X[:limit], Y[:limit]

    return X.as_matrix(), Y.as_matrix()
