"""
References:
    [1] Mhaskar: A direct approach for function approximation on data defined manifolds
"""

import numpy as np
from numpy.polynomial import polynomial
import pdb
from scipy.special import eval_hermite, factorial, gamma, loggamma

### Load Hermite polynomials

n_max = 2048

h_coefs = np.zeros((n_max + 1, n_max + 1))
h_coefs[0, 0] = np.pi ** (-1 / 4)
h_coefs[1, :2] = [0, np.sqrt(2) * (np.pi ** (-1 / 4))]

for k in np.arange(n_max - 1) + 2:
    hk = polynomial.polysub(
        np.sqrt(2 / k) * polynomial.polymulx(h_coefs[k - 1]),
        np.sqrt((k - 1) / k) * h_coefs[k - 2],
    )
    h_coefs[k, : len(hk)] = hk

### Functions


def hs(n, x, orthonormal=True):
    hs = eval_hermite(n, x)
    if orthonormal:
        denominator = (np.pi ** (1 / 4)) * (2 ** (n / 2)) * (factorial(n) ** (1 / 2))
        hs = hs / denominator
    return hs


def h(n, x, orthonormal=True):
    """
    Univariate Hermite polynomial of degree n, evaluated at points x.
    """
    h = polynomial.polyval(x, h_coefs[n])
    return h


def psi(n, x):
    """
    Univariate Hermite function of degree n, evaluated at points x.

    [1] equation (3.4)
    """
    H = h(n, x, orthonormal=True)
    weight = np.exp(-(x**2) / 2)
    psi = H * weight
    return psi


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


def P(m, q, x):
    """
    m, q:   integer parameters
    x:      points to evaluate P at

    [1] equation (3.5)
    """
    if q == 1:
        return (
            (np.pi ** (-1 / 4))
            * ((-1) ** m)
            * (np.exp(0.5 * loggamma(2 * m + 1) - (m * np.log(2) + loggamma(m + 1))))
            * psi(2 * m, x)
        )
    else:
        coef = 1 / (np.pi ** ((2 * q - 1) / 4) * gamma((q - 1) / 2))

        ls = np.arange(m + 1)
        coef_vec = np.power(np.zeros(m + 1) - 1, ls)
        coef_vec = coef_vec * np.exp(
            loggamma((q - 1) / 2 + m - ls) - loggamma(m - ls + 1)
        )
        coef_vec = coef_vec * np.exp(
            0.5 * loggamma(2 * ls + 1) - (ls * np.log(2) + loggamma(ls + 1))
        )

        coef_vec = np.repeat([coef_vec], len(x), axis=0).transpose()

        psis = np.zeros((m + 1, len(x)))
        for l in range(len(psis)):
            psis[l, :] = psi(2 * l, x)

        return coef * np.sum(coef_vec * psis, axis=0)


def Phi(n, q, x):
    """
    Hermite kernel of degree n, evaluated at points x. q is also a parameter.

    [1] equation (3.6)
    """
    lim = int(n**2 / 2)
    H = np.arange(lim + 1)
    H = hfuncinfty(np.sqrt(2 * H) / n)
    H = np.repeat([H], len(x), axis=0).transpose()

    Ps = np.zeros((lim + 1, len(x)))
    for m in range(len(Ps)):
        Ps[m, :] = P(m, q, x)

    return (H * Ps).sum(axis=0)
