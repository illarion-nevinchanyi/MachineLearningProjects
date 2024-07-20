import numpy as np
from sklearn.base import BaseEstimator


class KNearestNeighborsClassifier(BaseEstimator):
    def __init__(self, k=1):
        self.k = k
        self.X = None
        self.y = None
        self.classes = None     # a list of unique classes in our classification problem

    def fit(self, X, y):
        # TODO: Implement this method by storing X, y and infer the unique classes from y
        #       Useful numpy methods: np.unique
        self.X = X
        self.y = y
        self.classes = np.unique(self.y)

        return self

    def predict(self, X):
        # TODO: Predict the class labels for the data on the rows of X
        #       Useful numpy methods: np.argsort, np.argmax
        #       Broadcasting is really useful for this task.
        #       See https://numpy.org/doc/stable/user/basics.broadcasting.html

        predictions = np.zeros(X.shape[0], dtype=self.y.dtype)
        for i, x in enumerate(X):

            # Broadcasting
            distances = self.X - x
            distances **= 2
            distances = np.sum(distances, axis=1)
            distances = np.sqrt(distances)

            # or np.linalg
            # distances = np.linalg.norm(self.X - x, axis=1)

            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y[j] for j in k_indices]
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common = labels[np.argmax(counts)]
            predictions[i] = most_common

        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)