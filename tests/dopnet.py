"""Contains test functions for DopNET dataset"""

import os
import pdb
import pickle
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.filters import try_all_threshold, threshold_otsu
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    LeaveOneGroupOut,
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.svm import SVC

from src import hermite_classifier
from src.manifold_svm import SVDSVM, GrassmannSVM, find_best_hidden_dim
from src.preprocessing import Threshold, RadioFeatureExtractor

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def export_all_data_with_groups():
    """Exports all data of dopnet dataset with info on groups for cross-validation"""
    signals = None
    labels = None
    groups = None

    for person in ["A", "B", "C", "D", "E", "F"]:
        test = scipy.io.loadmat(
            os.path.realpath(
                "data/dopnet/Data_Per_PersonData_Training_Person_" + person + ".mat"
            )
        )

        # wave = 0, pinch = 1, swipe = 2, click = 3
        for gesture in [0, 1, 2, 3]:
            data = test["Data_Training"]["Doppler_Signals"][0, 0][0, gesture]

            for sample in range(len(data)):
                signal = data[sample, 0]

                if signals is None:
                    signals = [signal]
                    labels = [gesture]
                    groups = [person]
                else:
                    signals += [signal]
                    labels += [gesture]
                    groups += [person]

    labels = np.array(labels)
    groups = np.array(groups)

    pickle.dump(signals, open("data/dopnet/X.npy", "wb"))
    np.save("data/dopnet/y.npy", labels)
    np.save("data/dopnet/groups.npy", groups)

    print("exported data")


def import_all_data_with_groups(feature):
    """
    `feature` refers to Table I in Ritchie and Jones
        0: do nothing
        1: spectrogram summed intensity
        2: spectrogram variance
        3: spectrogram mean power (dB)
        4: spectrogram SVD -- summed singular values
        5: entropy of spectrogram intensity
    """
    X, y, groups = (
        np.load("data/dopnet/X.npy", allow_pickle=True),
        np.load("data/dopnet/y.npy"),
        np.load("data/dopnet/groups.npy"),
    )

    if feature == 0:
        pass
    elif feature == 1:
        for i, x in enumerate(X):
            X[i] = abs(x) / np.amax(abs(x))
    elif feature == 2:
        for i, x in enumerate(X):
            X[i] = 20 * np.log10(abs(x) / np.amax(abs(x)))
            X[i] -= X[i].mean()
            X[i] *= X[i]
    elif feature == 3:
        for i, x in enumerate(X):
            X[i] = 20 * np.log10(abs(x) / np.amax(abs(x)))
    elif feature == 4:
        raise ValueError(
            "4: spectrogram SVD -- summed singular values not implemented yet"
        )
    elif feature == 5:
        raise ValueError("5: entropy of spectrogram intensity not implemented yet")
    else:
        raise ValueError("Invalid feature")

    print(f"{datetime.now().strftime('%H:%M:%S')}: imported data\n")
    return X, y, groups


def cv_wrapper(estimator, X, y, splitters, params):
    """Performs and times many tests and saves results to file"""
    confusions = []
    confusions_pc = []
    scores = []
    train_times = []
    test_times = []
    label = datetime.now().strftime("%Y%m%d_%H%M%S_cv")

    print(f"{label}")
    print(params, "\n")

    for splitter in splitters:
        conf = []
        conf_pc = []
        sc = []
        train_ts = []
        test_ts = []

        for i, (train, test) in enumerate(splitter.split(X, y)):
            start_trial = time.time()
            estimator.fit([X[i] for i in train], y[train])
            train_ts.append(time.time() - start_trial)

            start_trial = time.time()
            pred = estimator.predict([X[i] for i in test])
            test_ts.append(time.time() - start_trial)

            confusion = confusion_matrix(y[test], pred)
            score = sum(pred == y[test]) / len(pred)
            conf.append(confusion)
            conf_pc.append(confusion / confusion.sum(axis=1)[:, None])
            sc.append(score)

            print(f"test size {splitter.test_size}, test {i + 1}/{splitter.n_splits}")
            print(f"score {score:.4f}")
            print(confusion)
            print(
                f"{datetime.now().strftime('%H:%M:%S')}: train {(train_ts[-1] / 60):.2f} mins, "
                f"test {(test_ts[-1] / 60):.2f} mins\n"
            )

        confusions.append(conf)
        confusions_pc.append(conf_pc)
        scores.append(sc)
        train_times.append(train_ts)
        test_times.append(test_ts)

    print(f"mean scores: {[round(sum(s) / len(s), 4) for s in scores]}")

    pickle.dump(
        {
            "confusions": confusions,
            "confusions_pc": confusions_pc,
            "scores": scores,
            "train_times": train_times,
            "test_times": test_times,
            "params": params,
        },
        open(f"tests/logs/{label}.pkl", "wb"),
    )


def signal_plots(estimator=None, unitise=False, normalise=False, savename=None):
    """Plot sample images of successful/unsuccessful signals from each class"""
    ind = None
    X, y, groups = import_all_data_with_groups(real=True)

    if ind is None:
        if estimator is None:
            estimator = SVDSVM(a=0.2, b=0.09, norm_sq=False)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.6)
        train, test = list(splitter.split(X, y))[0]
        estimator.fit(X[train], y[train])
        pred = estimator.predict(X[test])

        match_1 = list(np.logical_and(pred == y[test], y[test] == 0))
        match_3 = list(np.logical_and(pred == y[test], y[test] == 3))
        wrong_1 = list(np.logical_and(pred != y[test], y[test] == 0))
        wrong_3 = list(np.logical_and(pred != y[test], y[test] == 3))

        ind = [test[match_1.index(True)], test[wrong_1.index(True)]]
        # test[match_3.index(True)], test[wrong_3.index(True)]]
        print(ind)

    X = X[ind]
    thresh = Threshold(method="yen", unitise=unitise, normalise=normalise)
    X_prethresh = [np.copy(x) for x in X]
    X_thresh = np.array(thresh.transform(X), dtype=object)

    fig, axs = plt.subplots(2, len(ind))

    for row, ax_row in enumerate(axs):
        for col, ax in enumerate(ax_row):
            if row == 0:
                data = X_prethresh[col]
            else:
                data = X_thresh[col]
            im = ax.imshow(data, aspect="auto")

            if row == 0:
                title = f"Gesture {int(col / 2) * 3}"
                if col % 2 == 0:
                    title += ", match"
                else:
                    title += ", no match"
            else:
                title = "Post-threshold"

            ax.set_title(title)

    plt.tight_layout()

    if savename:
        plt.savefig(savename)
        print(f"saved {savename}")
    else:
        plt.show()


def test_split_sizes():
    params = {
        "feature": 3,  # Whether to convert data to real
        "threshold_method": "yen",  # Threshold method to use ("yen" or "otsu")
        "unitise": True,  # Whether to convert data to binary (overrides normalise if both are True)
        "normalise": False,  # Whether to normalise to [-1, 0]
        "test_sizes": [0.2, 0.4, 0.6, 0.8, 0.9],  # Test size ratios to use
        "n_splits": 5,  # Number of times to run each test size
    }

    # Import data (convert to real if needed)
    X, y, groups = import_all_data_with_groups(feature=params["feature"])
    hidden_dim = 10
    params["hidden_dim"] = hidden_dim

    estimator = hermite_classifier.HermiteClassifierOneHot(n=10, q=10)
    params["estimator"] = str(estimator)

    # Normalise/binary-ise data, then threshold
    thresh = Threshold(
        method=params["threshold_method"],
        unitise=params["unitise"],
        normalise=params["normalise"],
    )
    X = thresh.transform(X)

    # sklearn CV splitters to be used (allows different ratios to be tested)
    # StratifiedShuffleSplit preserves proportions of each class
    splitters = [
        StratifiedShuffleSplit(n_splits=params["n_splits"], test_size=ratio)
        for ratio in params["test_sizes"]
    ]
    # Call wrapper function to run/time/save the tests
    cv_wrapper(estimator, X, y, splitters, params)


def plot_radio_features():
    X, y, groups = import_all_data_with_groups(feature=0)
    rfe = RadioFeatureExtractor(features=[2, 3])
    X = rfe.transform(X)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].scatter(X[:, 0], X[:, 1], c=y, cmap="Accent", s=4)

    X = X[groups == "A"]
    y = y[groups == "A"]
    axs[1].scatter(X[:, 0], X[:, 1], c=y, cmap="Accent", s=4)

    plt.show()


def test_radio_features():
    X, y, groups = import_all_data_with_groups(feature=0)
    rfe = RadioFeatureExtractor(features=[2, 3])
    X = rfe.transform(X)
    X = X[groups == "A"]
    y = y[groups == "A"]

    # estimator = hermite_classifier.HermiteSVM(
    #     n=4, q=6, kern_gamma=0.2, hidden_dim=0, matrix_data=False
    # )
    # estimator = SVC()
    estimator = hermite_classifier.HermiteClassifierOneHot(n=10, q=10)
    # estimator = HermiteClassifier(n=1, q=1, alpha=1)
    splitters = [StratifiedShuffleSplit(n_splits=5, test_size=0.8)]
    params = {
        "feature": 3,  # whether to convert data to real
        "threshold_method": "yen",  # threshold method to use ("yen" or "otsu")
        "unitise": True,  # whether to convert data to binary (overrides normalise if both are True)
        "normalise": False,  # whether to normalise to [-1, 0]
        "test_sizes": [0.2, 0.4, 0.6, 0.8, 0.9],  # test size ratios to use
        "n_splits": 5,  # number of times to run each test size
    }
    cv_wrapper(estimator, X, y, splitters, params)


if __name__ == "__main__":
    # export_all_data_with_groups()
    test_split_sizes()
    # plot_radio_features()
    # test_radio_features()

    # signal_plots(estimator=SVDSVM(a=0.2, b=0.12, norm_sq=False), unitise=True,
    #         savename="20210326_002849_cv_examples.png")
    # signal_plots(estimator=SVDSVM(a=0.2, b=0.0042, norm_sq=True), unitise=True,
    #         savename="20210327_193305_cv_examples.png")
