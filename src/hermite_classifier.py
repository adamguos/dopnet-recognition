"""
References:
    [1] Mhaskar: A direct approach for function approximation on data defined manifolds
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import scipy
import sklearn
import time
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin

from src import localised_kernel


class HermiteClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier using Hermite approximator. For use in sklearn.pipeline.

    [1] equation (3.8)
    """
    def __init__(self, n, q):
        self.n = n
        self.q = q

    def fit(self, X_train, y_train):
        self.X_train = X_train

        self.le = sklearn.preprocessing.LabelEncoder()
        self.le.fit(y_train)
        self.y_train = self.le.transform(y_train)

    def predict(self, X_test):
        norms = np.zeros((len(self.X_train), len(X_test), len(self.X_train[0])))

        timer = time.time()
        for i, xtr in enumerate(self.X_train):
            norms[i, :] = xtr - X_test
        norms = np.linalg.norm(norms, axis=2)
        norms = norms / norms.max()
        print(f"euclidean norms: {time.time() - timer:.2f} secs, {norms.size} elems")

        timer = time.time()
        hermite = localised_kernel.eval_hermite(self.n**2, norms.flatten())
        print(f"hermite function: {time.time() - timer:.2f} secs, "
              "{hermite.size} elems, n^2 = {self.n ** 2}")

        timer = time.time()
        kernel = localised_kernel.eval_kernel(self.n, self.q, hermite)
        print(
            f"localised kernel: {time.time() - timer:.2f} secs, {kernel.size} elems, n = {self.n}")

        labels = self.y_train.reshape(-1, 1)
        pred = np.sum(labels * kernel.reshape(norms.shape), axis=0)
        # pred -= pred.min()
        pred *= len(self.le.classes_) / pred.max()
        classes = pred.astype(int)
        classes[classes == len(self.le.classes_)] = len(self.le.classes_) - 1
        pdb.set_trace()

        return self.le.inverse_transform(classes)

    def transform(self, X_test):
        return self.predict(X_test)


class HermiteClassifierOneHot(BaseEstimator, ClassifierMixin):
    """
    Classifier using Hermite approximator, using one hot encoder. For use in sklearn.pipeline.

    [1] equation (3.8)
    """
    def __init__(self,
                 n,
                 q,
                 gamma=0.8,
                 hidden_dim=10,
                 clenshaw=True,
                 save_norms=False,
                 load_norms=False,
                 verbose=True):
        self.n = n
        self.q = q
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.clenshaw = clenshaw
        self.save_norms = save_norms
        self.load_norms = load_norms
        self.verbose = verbose

    def fit(self, X_train, y_train):
        self.X_train = X_train

        self.onehot = sklearn.preprocessing.OneHotEncoder(sparse=False)
        self.y_train = self.onehot.fit_transform(y_train.reshape(-1, 1))

    def predict_proba(self, X_test):
        if self.load_norms:
            norms = np.load(self.load_norms)
        else:
            if self.X_train[0].ndim == 1:
                norms = self.vector_norms(self.X_train, X_test)
            elif self.X_train[0].ndim == 2:
                norms = self.svd_matrix_norms(self.X_train, X_test)
            else:
                raise ValueError("Data is in wrong shape (vector or matrix only)")

            if self.save_norms:
                np.save(self.save_norms, norms)
        norms *= self.gamma

        if self.clenshaw:
            timer = time.time()
            kernel = localised_kernel.eval_kernel_clenshaw(self.n, self.q, norms.flatten())
            if self.verbose:
                print(f"localised kernel: {time.time() - timer:.2f} secs, "
                      f"{kernel.size} elems, n = {self.n}, q = {self.q}")
        else:
            timer = time.time()
            hermite = localised_kernel.eval_hermite(self.n**2, norms.flatten())
            if self.verbose:
                print(f"hermite function: {time.time() - timer:.2f} secs, "
                      f"{hermite.size} elems, n^2 = {self.n ** 2}")

            timer = time.time()
            kernel = localised_kernel.eval_kernel(self.n, self.q, hermite)
            if self.verbose:
                print(f"localised kernel: {time.time() - timer:.2f} secs, "
                      f"{kernel.size} elems, n = {self.n}, q = {self.q}")

        pred_prob = np.zeros((len(X_test), len(self.onehot.categories_[0])))
        for category in range(len(self.onehot.categories_[0])):
            labels = self.y_train[:, category].reshape(-1, 1)
            pred_prob[:, category] = np.sum(labels * kernel.reshape(norms.shape),
                                            axis=0) / labels.sum()

        self.kernel = kernel.reshape(norms.shape)
        return pred_prob

    def predict(self, X_test):
        pred_prob = self.predict_proba(X_test)
        return self.predict_from_proba(pred_prob)

    def predict_from_proba(self, pred_prob):
        pred = np.ones_like(pred_prob)
        pred[pred_prob < pred_prob.max(axis=1).reshape(-1, 1)] = 0
        classes = self.onehot.inverse_transform(pred).flatten()

        return classes

    def significant(self, pred_prob, keep_ratio):
        part = -np.partition(-pred_prob, 1, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = part[:, 0] / part[:, 1]
        # ratios[np.nonzero(np.isinf(ratios))[0]] = 100

        indices = np.argsort(-ratios)[:int(len(ratios) * keep_ratio)]
        sig = np.zeros(len(ratios)).astype(bool)
        sig[indices] = True

        return sig

    def vector_norms(self, X_train, X_test):
        norms = np.zeros((len(self.X_train), len(X_test)), dtype="float32")

        timer = time.time()
        for i, xtr in enumerate(self.X_train):
            norms[i, :] = np.linalg.norm(xtr - X_test, axis=1)
        norms = norms / norms.max()
        if self.verbose:
            print(f"euclidean norms: {time.time() - timer:.2f} secs, {norms.size} elems")

        return norms

    def svd_matrix_norms(self, X_train, X_test):
        timer = time.time()
        svdu_tr = np.zeros((len(X_train), len(X_train[0]), self.hidden_dim))
        svdv_tr = np.zeros((len(X_train), self.hidden_dim))
        norms = np.zeros((len(self.X_train), len(X_test)))

        for i, x in enumerate(X_train):
            u, v, _ = np.linalg.svd(x, full_matrices=False)
            svdu_tr[i, :, :] = u[:, :self.hidden_dim]
            svdv_tr[i, :] = v[:self.hidden_dim]
        for i, x in enumerate(X_test):
            u, v, _ = np.linalg.svd(x, full_matrices=False)
            norms[:, i] = np.linalg.norm(svdu_tr - u[:, :self.hidden_dim], axis=(1, 2))
            # norms[:, i] += np.linalg.norm(svdv_tr - v[:self.hidden_dim], axis=1)
        norms = norms / norms.max()
        if self.verbose:
            print(f"svd norms: {time.time() - timer:.2f} secs, {norms.size} elems")

        return norms

    def grassmann_matrix_norms(self, X_train, X_test):
        timer = time.time()
        svd_tr = np.zeros((len(X_train), len(X_train[0]), self.hidden_dim))
        norms = np.zeros((len(self.X_train), len(X_test)))

        for i, x in enumerate(X_train):
            u, _, _ = np.linalg.svd(x, full_matrices=False)
            svd_tr[i, :, :] = u[:, :self.hidden_dim]
        for i, x in enumerate(X_test):
            u, _, _ = np.linalg.svd(x, full_matrices=False)
            matrix_prods = np.matmul(u[:, :self.hidden_dim].T, svd_tr)
            norms[:, i] = np.sqrt(
                np.abs(self.hidden_dim - np.linalg.norm(matrix_prods, axis=(1, 2))**2))
        norms = norms / norms.max()
        if self.verbose:
            print(f"grassmann norms: {time.time() - timer:.2f} secs, {norms.size} elems")

        return norms

    def transform(self, X_test):
        return self.predict(X_test)


class HermiteSVM(svm.SVC):
    """
    SVM classifier using Hermite localised kernel. For use in sklearn.pipeline.
    """
    def __init__(self, n, q, kern_gamma=0.8, verbose=False):
        self.n = n
        self.q = q
        self.kern_gamma = kern_gamma
        self.verbose = verbose

        def kern(x, y):
            dists = localised_kernel.get_pairwise_dists(x, y)
            dists = dists / dists.max()
            dists = dists * self.kern_gamma

            matrix = localised_kernel.eval_kernel_clenshaw(self.n, self.q, dists.flatten(),
                                                           self.verbose)
            matrix = matrix.reshape(dists.shape)

            return matrix

        super().__init__(kernel=kern, max_iter=100000)

    def transform(self, X):
        return super().predict(X)
