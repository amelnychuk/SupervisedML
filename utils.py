
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import warnings

class Data(object):

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, data):
        self._X = data

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

class DataSet(object):
    def __init__(self):
        self.train = Data()
        self.test = Data()

    @property
    def trainSet(self):
        return self.train.X, self.train.labels

    @property
    def testSet(self):
        return self.test.X, self.test.labels


class CONSTANTS(object):

    Mnist = DataSet()

    Mnist.train.X = pd.read_table("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/mnist_csv/Xtrain.txt",
                                  sep=',', header=None)
    Mnist.train.labels = pd.read_table("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/mnist_csv/label_train.txt",
                                       header=None, squeeze=True)

    Mnist.test.X = pd.read_table("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/mnist_csv/Xtest.txt",
                                 sep=',', header=None)

    Mnist.test.labels = pd.read_table("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/mnist_csv/label_test.txt",
                                      header=None, squeeze=True)

def get_data(dataSet, limit=None):
    """Gets mnist data from lazyprogrammers' hit hub.

    :param limit:
        (int) the maximum amount of samples your would like from mnist

    :return: np.ndarray (N,D), np.ndarray (N,)
        X, Y mnist features and labels
    """
    data, labels = dataSet

    X, Y = shuffle(data, labels)
    if limit is not None:
        xSize = len(X)
        if limit > xSize:
            warnings.warn("Limit of {} exceeds data sample size of {}".format(limit, xSize))
        X, Y = X[:limit], Y[:limit]

    return X.as_matrix(), Y.as_matrix()


a, b = get_data(CONSTANTS.Mnist.trainSet)