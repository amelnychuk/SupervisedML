
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import warnings

def show_digit(digit, label, num):
    plt.gray()
    plt.matshow(digit[num].reshape((8,8)))
    plt.title("Training: {}".format(label[num]))
    plt.show()



def get_data():
    """Gets mnist data from sklearn library. Shuffles and splits it

    :return: np.ndarray (N,D), np.ndarray (N,) np.ndarray (N,D), np.ndarray (N,)
        Xtrain (np.ndarray (N,D) ),
        Ytrain, (np.ndarray (N,) ),
        Xtest (np.ndarray (N,D) ),
        Ytest, (np.ndarray (N,) ),
    """
    data, targets = load_digits(return_X_y=True)

    X, Y = shuffle(data, targets)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=.1, train_size=.9)


    return Xtrain, Ytrain, Xtest, Ytest

