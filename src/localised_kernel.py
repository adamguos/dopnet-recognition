"""
References:
    [1] Mhaskar: A direct approach for function approximation on data defined manifolds
"""

import pdb

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import loggamma, gamma


def get_pairwise_dists(x, y):
    dists = cdist(x, y)
    return dists


def localised_kernel(x, y, n, q, gamma=1):
    norms = np.zeros((len(x), len(y)), dtype="float32")

    for i, xi in enumerate(x):
        norms[i, :] = np.linalg.norm(xi - y, axis=1)
    norms = norms / norms.max() * gamma

    return eval_kernel_clenshaw(n, q, norms.flatten())


def eval_hermite(n, x):
    """
    Evaluates Hermite function psi_k(x) = h_k(x) exp(-x^2/2) on all given x from degree 0 to n.
    Returns [a_{ij}], where a_{ij} = psi_i(x_j).

    [1] equation (3.3)
    """
    hermite = np.zeros((n + 1, len(x)))

    hermite[0, :] = np.pi**(-1 / 4) * np.exp(-x**2 / 2)
    hermite[1, :] = np.sqrt(2) * (np.pi**(-1 / 4)) * x * np.exp(-x**2 / 2)

    for k in np.arange(2, n + 1):
        print(f'k = {k}/{n}', end='\r')
        hermite[k, :] = np.sqrt(2 / k) * x * hermite[k - 1, :] - np.sqrt(
            (k - 1) / k) * hermite[k - 2, :]
    print()

    return hermite


def eval_kernel_old(n, q, hermite):
    """
    Localised kernel of degree n, evaluated on values of the Hermite function.

    [1] equation (3.6)
    """
    lim = int(n**2 / 2)

    H = np.arange(lim + 1)
    H = hfuncinfty(np.sqrt(2 * H) / n)
    H = np.repeat([H], hermite.shape[1], axis=0).transpose()

    Ps = np.zeros((lim + 1, hermite.shape[1]))
    for m in range(len(Ps)):
        Ps[m, :] = P(m, q, hermite)

    return (H * Ps).sum(axis=0)


def eval_kernel(n, q, hermite):
    """
    Localised kernel of degree n, evaluated on values of the Hermite function.

    [1] equation (3.6)
    """
    lim = max([int(n**2 / 2), 1])
    coefs = kernel_coef(n, q, lim)

    return np.sum(hermite[::2] * coefs.reshape(-1, 1), axis=0)


def eval_kernel_clenshaw(n, q, x, verbose=False):
    if verbose:
        print('Computing kernel...')

    lim = int(n**2 / 2)

    coefs = np.zeros((2, len(x)))
    if q > 1:
        kernel_coefs = kernel_coef(n, q, lim)
    else:
        kernel_coefs = np.ones(lim + 1)

    for j in reversed(range(1, lim + 1)):
        if verbose and j % 100 == 0:
            print(f'Evaluating kernel, {lim - j}/{lim}...', end='\r')

        next_coef = np.sqrt(2 / (j * (2 * j - 1))) * (x**2 - (4 * j - 3) / 2) * coefs[0]
        next_coef -= np.sqrt((j) * (2 * j - 1) / ((j + 1) * (2 * j + 1))) * coefs[1]
        # next_coef -= np.sqrt((j - 1) * (2*j - 3) / (j * (2*j - 1))) * coefs[1]
        next_coef += kernel_coefs[j]

        coefs[1] = coefs[0]
        coefs[0] = next_coef

    if verbose:
        print(' ' * 100, end='\r')
        print('Kernel computed')

    kernel = np.exp(-x**2 / 2) * (np.pi**(-1 / 4)) * coefs[0]
    kernel = kernel * np.sqrt(2) * (x**2 - 1 / 2)
    return kernel

    # if q == 1:
    #     m = np.arange(lim + 1)
    #     psi = eval_hermite(2 * lim, x)[::2, :]
    #     coefs = np.pi**(-1 / 4) * (-1)**m
    #     coefs *= np.exp(1 / 2 * loggamma(2 * m + 1) - 1 / 2 * np.log(2) - loggamma(m + 1))
    #     H = hfuncinfty(np.sqrt(2 * m / n))
    #     kernel = H[:, None] * (coefs[:, None] * psi)
    #     kernel = np.sum(kernel, axis=0)
    #     return kernel


def kernel_coef(n, q, lim):
    coefs = np.zeros((lim + 1, lim + 1))
    hermite_zero = eval_hermite(lim * 2, np.array([0]))

    for l in range(lim + 1):
        ind = np.arange(lim + 1 - l)
        coefs[l, :(lim + 1 - l)] = hfuncinfty(np.sqrt(2 * (l + ind)) / n)
        coefs[l, :(lim + 1 - l)] *= np.exp(loggamma((q - 1) / 2 + ind) - loggamma(ind + 1))
        coefs[l] *= hermite_zero[2 * l]

    coefs = coefs.sum(axis=1)
    coefs /= np.pi**((2 * q - 1) / 4) * gamma((q - 1) / 2)

    return coefs


# def localised_kernel(X, Y, n, q):
#     norms = np.zeros((len(X), len(Y), len(X[0])))
#     for i, x in enumerate(X):
#         norms[i, :] = x - Y
#     norms = np.linalg.norm(norms, axis=2)
#     norms = norms / norms.max()
#
#     hermite = eval_hermite(n ** 2, norms.flatten())
#     return eval_kernel(n, q, hermite).reshape(norms.shape)


def P(m, q, hermite):
    """
    m, q:   integer parameters
    x:      values of the Hermite function to evaluate P at

    [1] equation (3.5)
    """
    if q == 1:
        return (np.pi**(-1 / 4)) * (
            (-1)**m) * (np.exp(0.5 * loggamma(2 * m + 1) -
                               (m * np.log(2) + loggamma(m + 1)))) * hermite[2 * m, :]
    else:
        coef = 1 / (np.pi**((2 * q - 1) / 4) * gamma((q - 1) / 2))

        ls = np.arange(m + 1)
        coef_vec = np.power(np.zeros(m + 1) - 1, ls)
        coef_vec = coef_vec * np.exp(loggamma((q - 1) / 2 + m - ls) - loggamma(m - ls + 1))
        coef_vec = coef_vec * np.exp(0.5 * loggamma(2 * ls + 1) -
                                     (ls * np.log(2) + loggamma(ls + 1)))

        coef_vec = np.repeat([coef_vec], hermite.shape[1], axis=0).transpose()
        psis = hermite[:2 * (m + 1):2]

        return coef * np.sum(coef_vec * psis, axis=0)


def hfuncinfty(x):
    """
    x:  points to evaluate H at
    """
    y = np.zeros_like(x)
    x = np.abs(x)
    y[x <= 1 / 2] = 1

    ind = [i for i in range(len(x)) if 1 / 2 < x[i] and x[i] < 1]
    y[ind] = np.exp(-np.exp(2 / (1 - 2 * x[ind])) / (1 - x[ind]))

    return y
