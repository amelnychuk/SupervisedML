import numpy as np
from sortedcontainers import SortedList
from sklearn.metrics.pairwise import pairwise_distances

class KNN(object):
    def __init__(self, k):
        self._k = k

    def fit(self, Xtrain, Ytrain):
        self.X = Xtrain
        self.Y = Ytrain

    def predict(self, X):

        y = np.zeros(len(X))
        for i, x in enumerate(self.X):
            self.sl = SortedList()
            for j, xt in enumerate(X):
                diff = x - xt
                dist = diff.dot(diff)
                if len(self.sl) < self._k:
                    self.sl.add( (dist, self.Y[j]) )

                else:
                    if dist < self.sl[-1][0]:
                        del self.sl[-1]
                        self.sl.add( (dist, self.Y[j]) )

            #vote
            """
            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
            """
        #return y

    def predict(self, Xtest):

        N = len(Xtest)
        y = np.zeros(N)

        distances = pairwise_distances(Xtest, self.X)

        idx = distances.argsort(axis=1)[:, :self._k]

        votes = self.Y[idx]
        for i in range(N):
            y[i] = np.bincount(votes[i]).argmax()

        return y

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


