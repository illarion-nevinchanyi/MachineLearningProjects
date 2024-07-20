import numpy as np
from sklearn.base import BaseEstimator


def loss(w, b, C, X, y):
    # TODO: Implement the loss function (Equation 1)
    #       useful methods: np.sum, np.clip

    g = 1 - y * (np.einsum('ij, j', X, w) + b)              # g(w, b) = 1 - y_i(w^T * x_i + b)
    f_g = np.clip(g, 0, np.inf)                                          # f(g) = max(0, g)
    loss = 0.5 * np.einsum('i, i', w, w) + C * np.sum(f_g)

    return loss


def grad(w, b, C, X, y):
    # TODO: Implement the gradients of the loss with respect to w and b.
    #       Useful methods: np.sum, np.where, numpy broadcasting

    g = 1 - y * (np.einsum('ij, j', X, w) + b)
    indicator = np.where(g >= 0, 1, 0)
    grad_w, grad_b = w - C * np.einsum('ij, j', X.T, y * indicator), -C * np.sum(y * indicator)
    return grad_w, grad_b


class LinearSVM(BaseEstimator):
    def __init__(self, C=1, eta=1e-3, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.eta = eta

    def fit(self, X, y):
        # convert y such that components are not \in {0, 1}, but \in {-1, 1}
        y = np.where(y == 0, -1, 1)

        # TODO: Initialize self.w and self.b. Does the initialization matter?
        # doesnt matter
        # self.w = np.random.randint(0, 1000, X.shape[1])
        self.w = np.zeros(X.shape[1])   # X is (N x d)
        self.b = 0.0

        loss_list = []
        eta = self.eta  # starting learning rate
        for j in range(self.max_iter):
            # TODO: Compute the gradients, update self.w and self.b using `eta` as the learning rate.
            #       Compute the loss and add it to loss_list.
            grad_w, grad_b = grad(self.w, self.b, self.C, X, y)
            self.w = self.w - eta * grad_w
            self.b = self.b - eta * grad_b
            loss_list.append(loss(self.w, self.b, self.C, X, y))

            # decaying learning rate
            eta = eta * 0.99

        return loss_list

    def predict(self, X):
        # TODO: Predict class labels of unseen data points on rows of X
        #       NOTE: The output should be a vector of 0s and 1s (*not* -1s and 1s)
        condition = np.einsum('ij, j', X, self.w) + self.b >= 0
        y_pred = np.where(condition, 1, 0)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
