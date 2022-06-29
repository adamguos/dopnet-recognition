"""
References:
    [1] Jayasumana et al: Kernel methods on Riemannian manifolds with Gaussian RBF kernels
"""

from time import perf_counter
import pdb

import numpy as np
from sklearn import svm

from src.hermite import Phi


def frobenius_kernel(Y1,
                     Y2,
                     alpha,
                     beta,
                     gamma,
                     hidden_dim,
                     include_v=False,
                     norm_sq=False,
                     skip_exp=False):
    """
    Computes a kernel using the Frobenius norm on the singular value decompositions of multi-feature
    signals in Y1 and Y2.
    """
    tic = perf_counter()

    # Compute SVD of Y1 and Y2 and truncate at hidden_dim
    U_Y1, S_Y1, _ = np.linalg.svd(Y1, full_matrices=False)
    U_Y2, S_Y2, _ = np.linalg.svd(Y2, full_matrices=False)
    U_Y1, U_Y2 = U_Y1[:, :, :hidden_dim], U_Y2[:, :, :hidden_dim]
    S_Y1, S_Y2 = S_Y1[:, :hidden_dim], S_Y2[:, :hidden_dim]
    # svd_Y1, svd_Y2 = [], []
    # U_Y1 = np.zeros((len(Y1), Y1[0].shape[0], min(hidden_dim, Y1[0].shape[1])))
    # S_Y1 = np.zeros((len(Y1), min(hidden_dim, Y1[0].shape[1])))
    # V_Y1 = np.zeros((len(Y1), min(hidden_dim, Y1[0].shape[1]), Y1[0].shape[1]))
    # U_Y2 = np.zeros((len(Y2), Y2[0].shape[0], min(hidden_dim, Y2[0].shape[1])))
    # S_Y2 = np.zeros((len(Y2), min(hidden_dim, Y2[0].shape[1])))
    # V_Y2 = np.zeros((len(Y2), min(hidden_dim, Y2[0].shape[1]), Y2[0].shape[1]))
    # for i, y in enumerate(Y1):
    # U, S, V = np.linalg.svd(y, full_matrices=False)
    # U = U[:, :hidden_dim]
    # s = s[:hidden_dim]
    # Vh = Vh[:hidden_dim, :]
    # svd_Y1.append((U, s, Vh))
    # U_Y1[i] = U[:, :hidden_dim]
    # S_Y1[i] = S[:hidden_dim]
    # V_Y1[i] = V[:hidden_dim, :]
    # for i, y in enumerate(Y2):
    # U, S, V = np.linalg.svd(y, full_matrices=False)
    # U = U[:, :hidden_dim]
    # s = s[:hidden_dim]
    # Vh = Vh[:hidden_dim, :]
    # svd_Y2.append((U, s, Vh))
    # U_Y2[i] = U[:, :hidden_dim]
    # S_Y2[i] = S[:hidden_dim]
    # V_Y2[i] = V[:hidden_dim, :]

    print(f'singular value decompositions: {perf_counter() - tic:0.4f} secs')

    # Compute kern(x, y) for all x \in X, y \in Y
    # Store into array of size (len(X), len(Y))
    if norm_sq:
        kern_matrix = beta * np.linalg.norm(S_Y1[:, None] - S_Y2, axis=2) ** 2 + \
                alpha * np.linalg.norm(U_Y1[:, None] - U_Y2, axis=(2, 3)) ** 2
    else:
        kern_matrix = beta * np.linalg.norm(S_Y1[:, None] - S_Y2, axis=2)
        # alpha * np.linalg.norm(U_Y1[:, None] - U_Y2, axis=(2, 3))

        U_norm = np.zeros((len(Y1), len(Y2)))
        U_norm[:len(U_Y1) // 2, :len(U_Y2) // 2] += np.linalg.norm(U_Y1[:len(U_Y1) // 2, None] -
                                                                   U_Y2[:len(U_Y2) // 2],
                                                                   axis=(2, 3))
        U_norm[len(U_Y1) // 2:, :len(U_Y2) // 2] += np.linalg.norm(U_Y1[len(U_Y1) // 2:, None] -
                                                                   U_Y2[:len(U_Y2) // 2],
                                                                   axis=(2, 3))
        U_norm[:len(U_Y1) // 2, len(U_Y2) // 2:] += np.linalg.norm(U_Y1[:len(U_Y1) // 2, None] -
                                                                   U_Y2[len(U_Y2) // 2:],
                                                                   axis=(2, 3))
        U_norm[len(U_Y1) // 2:, len(U_Y2) // 2:] += np.linalg.norm(U_Y1[len(U_Y1) // 2:, None] -
                                                                   U_Y2[len(U_Y2) // 2:],
                                                                   axis=(2, 3))
        kern_matrix += alpha * U_norm

    # if include_v:
    #     kern_matrix += gamma * np.linalg.norm(V_Y1[:, None] - V_Y2, axis=(2, 3))

    # kern_matrix = np.empty((len(Y1), len(Y2)))
    # kern_matrix[:] = np.nan
    # for i in range(len(Y1)):
    #     for j in range(len(Y2)):
    #         U1, S1, V1 = svd_Y1[i]
    #         U2, S2, V2 = svd_Y2[j]

    #         if norm_sq:  # Gaussian kernel
    #             kern_matrix[i, j] = beta * (np.linalg.norm(S1 - S2) ** 2) \
    #                 + alpha * (np.linalg.norm(U1 - U2) ** 2)
    #         else:  # Laplace kernel
    #             kern_matrix[i, j] = beta * (np.linalg.norm(S1 - S2)) \
    #                 + alpha * (np.linalg.norm(U1 - U2))

    #         if include_v:  # include right singular vectors if requested
    #             kern_matrix[i, j] += gamma * np.linalg.norm(V1[:V2.shape[0], :V2.shape[1]] -
    #                                                         V2[:V1.shape[0], :V1.shape[1]])

    if not skip_exp:
        kern_matrix = np.exp(-kern_matrix)

    print(f'frobenius norm: {perf_counter() - tic:0.4f} secs')

    return kern_matrix


def projection_kernel(X, Y, kern_gamma, hidden_dim, skip_exp=False):
    svd_X, svd_Y = [], []
    for i, x in enumerate(X):
        U, _, _ = np.linalg.svd(x, full_matrices=False)
        svd_X.append(U[:, :hidden_dim])
    for i, y in enumerate(Y):
        U, _, _ = np.linalg.svd(y, full_matrices=False)
        svd_Y.append(U[:, :hidden_dim])

    kern_matrix = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            kern_matrix[i,
                        j] = kern_gamma * (hidden_dim - (np.linalg.norm(svd_X[i].T @ svd_Y[j])**2))

    if not skip_exp:
        kern_matrix = np.exp(-kern_matrix)

    return kern_matrix


def find_best_hidden_dim(data, norm_threshold=0.95):
    """
    Finds smallest number of singular values/vectors that preserves `norm_threshold` proportion of
    Frobenius norm of all points in data.
    """
    max_index = 0

    for x in data:
        U, S, V = np.linalg.svd(x, full_matrices=False)

        total_norm = np.linalg.norm(x)
        # singular_sums[i] = sqrt of sum of squares of first (i + 1) singular values
        singular_sums = [np.sqrt(np.square(S[:i]).sum()) for i in range(1, len(S) + 1)]

        # Get index of first elem of singular_sums that reaches norm_threshold * total_norm
        index = np.where(singular_sums >= norm_threshold * total_norm)[0][0]
        max_index = max(max_index, index)

    print(f"final max index = {max_index}")
    return max_index


class GrassmannSVM(svm.SVC):
    """
    SVM classifier using Gaussian kernel and projection metric on Grassmann manifold.
    Intended to not be used with ARMA parametrisation.
    """
    def __init__(self, kern_gamma=0.2, hidden_dim=10, probability=False):
        self.kern_gamma = kern_gamma
        self.hidden_dim = hidden_dim
        self.probability = probability

        # sklearn expects kern to be a function taking two arrays of data X and Y,
        # returning the value of kern(x, y) at each x \in X, y \in Y
        # should return an array of floats, shape (len(X), len(Y))
        def kern(X, Y):
            svd_X, svd_Y = [], []
            for i, x in enumerate(X):
                U, _, _ = np.linalg.svd(x, full_matrices=False)
                svd_X.append(U[:, :self.hidden_dim])
            for i, y in enumerate(Y):
                U, _, _ = np.linalg.svd(y, full_matrices=False)
                svd_Y.append(U[:, :self.hidden_dim])

            kern_matrix = np.zeros((len(X), len(Y)))
            for i in range(len(X)):
                for j in range(len(Y)):
                    kern_matrix[i, j] = np.exp(-self.kern_gamma *
                                               (self.hidden_dim -
                                                (np.linalg.norm(svd_X[i].T @ svd_Y[j])**2)))

            return kern_matrix

        super().__init__(kernel=kern, probability=self.probability)

    def transform(self, X):
        return super().predict(X)


class SVDSVM(svm.SVC):
    """
    SVM classifier using Frobenius norm on singular value decompositions between multi-feature
    signals. For use in sklearn.pipeline.
    """
    def __init__(self,
                 a=0.1,
                 b=0.1,
                 c=None,
                 hidden_dim=10,
                 scaled=False,
                 include_v=False,
                 norm_sq=False,
                 probability=False):
        self.a = a
        self.b = b
        self.c = a if c is None else c
        self.hidden_dim = hidden_dim
        self.scaled = scaled
        self.include_v = include_v
        self.norm_sq = norm_sq
        self.probability = probability

        def kern(Y1, Y2):
            return frobenius_kernel(Y1,
                                    Y2,
                                    self.a,
                                    self.b,
                                    self.c,
                                    hidden_dim=self.hidden_dim,
                                    include_v=self.include_v,
                                    norm_sq=self.norm_sq)

        super().__init__(kernel=kern, probability=self.probability)

    def transform(self, X):
        return super().predict(X)
