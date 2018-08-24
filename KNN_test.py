import numpy as np
import matplotlib.pyplot as plt
from utils import get_data
from KNN import KNN



Xtrain, Ytrain, Xtest, Ytest = get_data()
train_scores = []
test_scores = []
ks = np.arange(1,11)
for k in ks:
    print "\nk : {}".format(k)
    knn = KNN(k)
    knn.fit(Xtrain, Ytrain)
    train_scores.append(knn.score(Xtrain, Ytrain))
    test_scores.append(knn.score(Xtest, Ytest))

plt.plot(ks, train_scores, label="Train")
plt.plot(ks, test_scores, label="Test")
plt.legend()
plt.show()