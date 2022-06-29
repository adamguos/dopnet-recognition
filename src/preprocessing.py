import os
import pdb

import numpy as np
import pandas as pd
import scipy.signal
from scipy.io import wavfile, loadmat
from skimage.filters import threshold_otsu, threshold_yen
from sklearn.base import BaseEstimator
import sklearn.decomposition


class Trimmer(BaseEstimator):
    """
    Trims time series data. Expects timesteps to span axis 1. For use in sklearn.pipeline.

    Parameters:
    start, end: indices to start and end trim, supports Python rules. None means no trim.
    """
    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        trimmed = []
        for x in X:
            trimmed.append(x[self.start:self.end])
        return trimmed


class Average(BaseEstimator):
    """
    Averages across columns. Expects timesteps to span axis 1. For use in sklearn.pipeline.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        averaged = []
        for x in X:
            averaged.append(x.mean(axis=1))
        return averaged


class Flatten(BaseEstimator):
    """
    Flattens 3D signals into 2D. Expects timesteps to span axis 1. For use in sklearn.pipeline.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        flattened = []
        for x in X:
            flattened.append(x.flatten())
        return flattened


class PCA(BaseEstimator):
    """
    Performs PCA on supplied signals. Expects timesteps to span axis 1. For use in sklearn.pipeline.
    """
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pcad = []
        for x in X:
            x = x.transpose()
            pca = sklearn.decomposition.PCA(n_components=self.n_components)
            pca.fit(x)
            pcad.append(pca.transform(x))
        return pcad


class Resize(BaseEstimator):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        for i in range(len(X)):
            X[i] = cv2.resize(X[i], dsize=(self.x, self.y), interpolation=cv2.INTER_CUBIC)
        return np.array(X)


class Threshold(BaseEstimator):
    """
    Applies various methods to zero-out values below threshold. For use in sklearn.pipeline.
    """
    def __init__(self, method="yen", normalise=False, unitise=False):
        self.normalise = normalise
        self.unitise = unitise
        methods = ["otsu", "yen"]

        if method == "otsu":
            self.threshold = threshold_otsu
        elif method == "yen":
            self.threshold = threshold_yen
        else:
            raise ValueError(f"Threshold expected one of: {methods}:")
        self.method = method

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        # Xp = []
        for i, _ in enumerate(X):
            if X[i].dtype == "complex128":
                thresh = self.threshold(np.abs(X[i]))
                X[i][np.abs(X[i]) < thresh] = 0
            else:
                thresh = self.threshold(X[i])
                X[i][X[i] < thresh] = 0

            if self.unitise:
                magnitudes = np.abs(X[i])
                magnitudes[magnitudes == 0] = 1
                X[i] = X[i] / magnitudes
            elif self.normalise:
                X[i] = X[i] / np.amax(np.abs(X[i]))

            # Xp.append(x)

        # return Xp
        return X


class RadioFeatureExtractor(BaseEstimator):
    """
    Extracts several features from radio signals. Each signal is represented by a complex-valued
    matrix where each row is a feature, each column is a timestep.

    Features from Table I of Ritchie and Jones, "Micro-Doppler Gesture Recognition using Doppler,
    Time and Range Based Features":
        1. spectrogram summed intensity
        2. spectrogram variance
        3. spectrogram mean power (dB)
        4. spectrogram SVD -- summed singular values
        5. entropy of spectrogram intensity
    """
    def __init__(self, features=[1, 2, 3, 4]):
        self.features = features

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        result = np.zeros((len(X), len(self.features)))

        for signal_i, x in enumerate(X):
            row = []
            for feature in self.features:
                row.append(self.get_feature(x, feature))
            result[signal_i, :] = np.array(row)

        return result

    def get_feature(self, x, i):
        if i == 1:
            return self.summed_intensity(x)
        elif i == 2:
            return self.variance(x)
        elif i == 3:
            return self.mean_power(x)
        elif i == 4:
            return self.summed_singular_values(x)
        elif i == 5:
            raise ValueError("Feature type 5: entropy of spectrogram intensity not implemented yet")
        else:
            raise ValueError(f"Invalid feature type: {i}")

    def summed_intensity(self, x):
        return np.abs(x).sum()

    def variance(self, x):
        return np.sum((20 * np.log10(np.abs(x)) - self.mean_power(x))**2) / x.size

    def mean_power(self, x):
        return 20 / x.size * np.log10(np.abs(x)).sum()

    def summed_singular_values(self, x):
        Sigma = np.linalg.svd(x, compute_uv=False)
        return Sigma.sum()
