"""Contains test functions for DopNET dataset"""

import os
import pdb
import pickle
import re
import sys
import time
from datetime import datetime
from types import SimpleNamespace

sys.path.append("../")
sys.path.append("../src")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.io
from skimage.filters import try_all_threshold, threshold_otsu, threshold_yen
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
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from torch.autograd import Variable
import torch
import torch.nn as nn

from mpl_toolkits import mplot3d

from models import (
    conv3Network,
    conv7Network,
    SVDSVM,
    GrassmannSVM,
    train_Model,
    test_Model,
    RManSVM,
    ProdGrassmannSVM,
)
from data import pad_data, process_data
from plotting import (
    make_single_person_plots,
    plot_class_size_results,
    print_result_table,
)
from hermite_classifier import HermiteSVM

import logging

################################################################################

logger = logging.getLogger(__name__)


def SetLoggingLevel(level):
    """
    This sets the logging level of the logger in this module.
    """
    handler = logging.StreamHandler()
    if level == "DEBUG":
        logger.setLevel("DEBUG")
        handler.setLevel(logging.DEBUG)
    else:
        logger.setLevel("INFO")
        handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # create file handler which logs even debug messages
    fh = logging.FileHandler("debug2.log")
    logger.addHandler(fh)


################################################################################

BASE_DIR = os.getcwd()

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

# device = torch.device("cuda:0")
device = torch.device("cpu")


def run_KSVM_trial(estimator, X_train, X_test, y_train, y_test, name):
    """ """
    start_trial = time.time()
    estimator.fit(X_train, y_train)
    train_ts = time.time() - start_trial
    start_trial = time.time()
    pred = estimator.predict(X_test)
    test_ts = time.time() - start_trial
    acc = np.where((pred - y_test) == 0)[0].shape[0] / len(y_test)
    logger.info("SVM {} acc: {}".format(name, acc))
    # pdb.set_trace()
    return acc, test_ts, train_ts


def run_CNN_trial(X_train, X_test, y_train, y_test, max_time_dim, name, Nfilts, Nsize):
    """ """
    Nepochs = 50  # should be passed in
    X_train_, data_dim = pad_data(X_train, max_time_dim)
    X_test_, data_dim = pad_data(X_test, max_time_dim)
    start_trial = time.time()
    model = train_Model(X_train_, y_train, Nepochs, Nfilts, Nsize)
    train_ts = time.time() - start_trial
    start_trial = time.time()
    acc = test_Model(X_test_, y_test, model)
    test_ts = time.time() - start_trial
    logger.info("CNN {} acc: {}".format(name, acc))
    return acc, test_ts, train_ts


def run_PCA_KNN_trial(X_train, X_test, y_train, y_test, max_time_dim, name, pca_n=30):
    from sklearn.pipeline import make_pipeline
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.decomposition import PCA

    estimator = make_pipeline(
        PCA(n_components=pca_n, random_state=42), KNeighborsClassifier(n_neighbors=5)
    )
    start_trial = time.time()
    X_train, data_dim = pad_data(X_train, max_time_dim)
    X_test, data_dim = pad_data(X_test, max_time_dim)
    # X_train = np.stack([x.flatten() for x in X_train], axis=0)
    # X_test = np.stack([x.flatten() for x in X_test], axis=0)
    # X_train = [x.flatten() for x in X_train]
    # X_test = [x.flatten() for x in X_test]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    estimator.fit(X_train, y_train)
    train_ts = time.time() - start_trial
    start_trial = time.time()
    pred = estimator.predict(X_test)
    test_ts = time.time() - start_trial
    acc = np.where((pred - y_test) == 0)[0].shape[0] / len(y_test)
    logger.info("PCA_KNN {} acc: {}".format(name, acc))
    return acc, test_ts, train_ts


def run_LocSVM_trial(
    X_train, X_test, y_train, y_test, max_time_dim, n, q, name, pca_n=30
):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline

    estimator = make_pipeline(
        PCA(n_components=pca_n, random_state=42), HermiteSVM(n, q, verbose=True)
    )
    start_trial = time.time()
    X_train, data_dim = pad_data(X_train, max_time_dim)
    X_test, data_dim = pad_data(X_test, max_time_dim)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    estimator.fit(X_train, y_train)
    train_ts = time.time() - start_trial
    start_trial = time.time()
    pred = estimator.predict(X_test)
    test_ts = time.time() - start_trial
    acc = np.where((pred - y_test) == 0)[0].shape[0] / len(y_test)
    logger.info("LocSVM {} acc: {}".format(name, acc))
    return acc, test_ts, train_ts


def run_all_trial(
    X_train,
    X_test,
    y_train,
    y_test,
    max_time_dim=None,
    mode="normalize",
    base_dir="None",
    trial=None,
):
    """ """
    logger.info("Begin running trials")
    logger.info("Make experiment result directory")
    if (base_dir is not None) and (trial is not None):
        exp_path = "{}_{}".format(mode, trial)
        file_path = os.path.join(base_dir, exp_path)
        save = True
        logger.info(f"Base path: {file_path}")
    else:
        save = False

    def save_results(results, name):
        try:
            if not os.path.exists(file_path):
                os.mkdir(file_path)
            file_name = os.path.join(file_path, "{}.pkl".format(name))
            logger.info(
                "saving results for file_name: {} (method: {})".format(file_name, name)
            )
            with open(file_name, "wb") as file:
                pickle.dump(results, file)
        except:
            # need to add control for specific conditions
            print("Could not save file {}".format(file_path))
        return

    name = "PCA_KNN"
    if os.path.exists(os.path.join(file_path, f"{name}.pkl")):
        logger.info(f"Skipping {name}")
        with open(os.path.join(file_path, f"{name}.pkl"), "rb") as file:
            PCA_KNN_results = pickle.load(file)
    else:
        logger.info(
            "Running PCA_KNN {} - parameters: hidden_dim={}, K = {}".format(mode, 30, 5)
        )
        PCA_KNN_results = run_PCA_KNN_trial(
            X_train, X_test, y_train, y_test, max_time_dim, "PCA_KNN"
        )
        logger.info("PCA_KNN results: {}".format(PCA_KNN_results))
        if save:
            save_results(PCA_KNN_results, name)

        # PCA_KNN_results = (0, 0, 0)

    name = "SVDG"
    if os.path.exists(os.path.join(file_path, f"{name}.pkl")):
        logger.info(f"Skipping {name}")
        with open(os.path.join(file_path, f"{name}.pkl"), "rb") as file:
            SVDG_results = pickle.load(file)
    else:
        trial_tic = time.time()
        logger.info("Trial start time: {} ({})".format(time.strftime("%c"), trial_tic))
        ## SVD Gaussian
        if mode == "binary":
            hidden_dim = 4
            a = 0.2
            b = 0.0042
        elif mode == "normalize":
            hidden_dim = 3
            a = 0.2
            b = 0.0042
        logger.info(
            "Running SVDG {} - parameters: hidden_dim={}, a = {}, b = {}".format(
                mode, hidden_dim, a, b
            )
        )
        estimator_SVDG = SVDSVM(
            a=a, b=b, hidden_dim=hidden_dim, norm_sq=True, max_iter=5000
        )
        # acc_SVDG, test_ts_SVDG, train_ts_SVDG = run_KSVM_trial(estimator_SVDG, X_train, X_test,
        #                                                        y_train, y_test, 'SVDG')
        SVDG_results = run_KSVM_trial(
            estimator_SVDG, X_train, X_test, y_train, y_test, "SVDG"
        )
        logger.info("SVDG results: {}".format(SVDG_results))
        if save:
            # save_results(SVDG_results, os.path.join(file_path,'SVDG'))
            save_results(SVDG_results, name)

    ## Grassman Gaussian
    name = "GG"
    if os.path.exists(os.path.join(file_path, f"{name}.pkl")):
        logger.info(f"Skipping {name}")
        with open(os.path.join(file_path, f"{name}.pkl"), "rb") as file:
            GG_results = pickle.load(file)
    else:
        if mode == "binary":
            hidden_dim = 5
            gamma = 0.2
        elif mode == "normalize":
            hidden_dim = 7
            gamma = 0.2
        logger.info(
            "Running GG {} - parameters: hidden_dim = {}, gamma = {}".format(
                mode, hidden_dim, gamma
            )
        )
        estimator_GG = GrassmannSVM(kern_gamma=gamma, hidden_dim=hidden_dim)
        # acc_GG, test_ts_GG, train_ts_GG = run_KSVM_trial(estimator_GG, X_train, X_test, y_train,
        #                                                  y_test, 'GG')
        GG_results = run_KSVM_trial(
            estimator_GG, X_train, X_test, y_train, y_test, "GG"
        )
        logger.info("GG results: {}".format(GG_results))
        if save:
            save_results(GG_results, name)

    ## SVD Laplace
    name = "SVDL"
    if os.path.exists(os.path.join(file_path, f"{name}.pkl")):
        logger.info(f"Skipping {name}")
        with open(os.path.join(file_path, f"{name}.pkl"), "rb") as file:
            SVDL_results = pickle.load(file)
    else:
        if mode == "binary":
            hidden_dim = 11
            a = 0.2
            b = 0.12
        elif mode == "normalize":
            hidden_dim = 19
            a = 0.2
            b = 0.12
        logger.info(
            "Running SVDL {} - parameters: hidden_dim={}, a = {}, b = {}".format(
                mode, hidden_dim, a, b
            )
        )
        estimator_SVDL = SVDSVM(a=0.2, b=0.12, hidden_dim=hidden_dim, norm_sq=False)
        SVDL_results = run_KSVM_trial(
            estimator_SVDL, X_train, X_test, y_train, y_test, "SVDL"
        )
        logger.info("SVDL results: {}".format(SVDL_results))
        if save:
            save_results(SVDL_results, name)

    ## neural netowrk
    name = "CNN1"
    if os.path.exists(os.path.join(file_path, f"{name}.pkl")):
        logger.info(f"Skipping {name}")
        with open(os.path.join(file_path, f"{name}.pkl"), "rb") as file:
            CNN1_results = pickle.load(file)
    else:
        Nfilts = [8, 16, 32]
        Nsize = [20, 10, 5]
        logger.info("Running CNN1 {} - parameters: Nfilts & Nsize")
        logger.info(f"filts & sizes: {Nfilts}, {Nsize}")
        CNN1_results = run_CNN_trial(
            X_train, X_test, y_train, y_test, max_time_dim, "moodel-1", Nfilts, Nsize
        )
        logger.info("CNN1 results: {}".format(CNN1_results))
        if save:
            save_results(CNN1_results, "CNN1")

    name = "CNN2"
    if os.path.exists(os.path.join(file_path, f"{name}.pkl")):
        logger.info(f"Skipping {name}")
        with open(os.path.join(file_path, f"{name}.pkl"), "rb") as file:
            CNN2_results = pickle.load(file)
    else:
        Nfilts = [5, 4, 2]
        Nsize = [5, 5, 5]
        logger.info("Running CNN2 {} - parameters: Nfilts & Nsize")
        logger.info(f"filts & sizes: {Nfilts}, {Nsize}")
        CNN2_results = run_CNN_trial(
            X_train, X_test, y_train, y_test, max_time_dim, "moodel-2", Nfilts, Nsize
        )
        logger.info("CNN2 results: {}".format(CNN2_results))
        if save:
            save_results(CNN2_results, "CNN2")

    # high-dimensional localised kernel SVM
    name = "LocSVM16"
    if os.path.exists(os.path.join(file_path, f"{name}.pkl")):
        logger.info(f"Skipping {name}")
        with open(os.path.join(file_path, f"{name}.pkl"), "rb") as file:
            LocSVM16_results = pickle.load(file)
    else:
        n = 16
        q = 2
        logger.info(f"Running {name} - parameters: n & q")
        logger.info(f"n = {n}, q = {q}")
        LocSVM16_results = run_LocSVM_trial(
            X_train, X_test, y_train, y_test, max_time_dim, n, q, name
        )
        logger.info(f"{name} results: {LocSVM16_results}")
        if save:
            save_results(LocSVM16_results, name)

    name = "LocSVM64"
    if os.path.exists(os.path.join(file_path, f"{name}.pkl")):
        logger.info(f"Skipping {name}")
        with open(os.path.join(file_path, f"{name}.pkl"), "rb") as file:
            LocSVM64_results = pickle.load(file)
    else:
        if True:
            n = 64
            q = 18
            logger.info(f"Running {name} - parameters: n & q")
            logger.info(f"n = {n}, q = {q}")
            LocSVM64_results = run_LocSVM_trial(
                X_train, X_test, y_train, y_test, max_time_dim, n, q, name
            )
            logger.info(f"{name} results: {LocSVM64_results}")
            if save:
                save_results(LocSVM64_results, name)
        else:
            LocSVM64_results = (0, 0, 0)

    # Nfilts = [16, 32, 64, 128, 128, 128, 128 ]
    # Nsize = [7, 5, 5, 3, 3, 3, 3]
    # logger.info("Running CNN3 {} - parameters: Nfilts & Nsize")
    # logger.info('filts & sizes: ', Nfilts,  Nsize)
    # CNN3_results = run_CNN_trial(X_train, X_test, y_train, y_test, max_time_dim, 'moodel-3',
    #                              Nfilts, Nsize)
    # logger.info('CNN3 results: {}'.format(CNN3_results))
    # if save:
    #     save_results(CNN3_results, 'CNN3')
    # trial_time = trial_tic-time.time()

    # logger.info("Trial time is {}.format(trial_time)")
    # logger.info("Completed Trials")
    # logger.info("--------------------------------------------")
    return (
        SVDG_results,
        GG_results,
        SVDL_results,
        CNN1_results,
        CNN2_results,
        PCA_KNN_results,
        LocSVM16_results,
        LocSVM64_results,
    )


def evaluate_test_models(Ntrials, mode, test_size, base_dir):

    #### NEED TO FINISH THIS FUNCTION... HC FETURES AND CNN ARE NOT SET UP AND NO VARIABLES TO
    #### STORE THINGS DEFINED ABOVE

    ## create splits of train and test data
    # splitter = StratifiedShuffleSplit(n_splits=Ntrials, test_size=0.2)
    # pdb.set_trace()
    # train, test = list(splitter.split(X, y))[0]
    # Nepochs = 150
    Nepochs = 100
    hidden_dim = 5

    ## load data
    logger.info("Loading Data")
    X = np.load("../data/dopnet/X.npy", allow_pickle=True)
    y = np.load("../data/dopnet/y.npy")
    groups = np.load("../data/dopnet/groups.npy")
    max_time_dim = max([x.shape[1] for x in X])

    logger.info("Process data and run trials")
    results = []
    cnt = 0
    for cnt in range(Ntrials):  # cnt = cnt
        # if cnt >=3:
        print("Trial: ", cnt)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True
        )
        logger.info(
            (
                "Trial summary (intial)): trial {} - X_train length = {} - y_train length = {}"
                + " - X_test length = {} - y_test length = {}"
            ).format(cnt, len(X_train), len(y_train), len(X_test), len(y_test))
        )
        if False:
            X_train = process_data(X_train, mode)
            X_test = process_data(X_test, mode)
        result = run_all_trial(
            X_train, X_test, y_train, y_test, max_time_dim, mode, base_dir, cnt
        )  # trial = cnt
        results.append(result)
        logger.info("Trial {} complete".format(cnt))

    def calc_statistics(results, stat_idx):
        num_methods = len(results[0])
        num_trials = len(results)
        vals = np.zeros((num_methods, num_trials))
        for ii in range(num_methods):
            for jj in range(num_trials):
                vals[ii, jj] = results[jj][ii][stat_idx]
        av = np.mean(vals, axis=1)
        var = np.var(vals, axis=1)
        return av, var

    ### modified
    acc_stats = calc_statistics(results, 0)
    test_ts_stats = calc_statistics(results, 1)
    train_ts_stats = calc_statistics(results, 2)
    logger.info("Results calculated - exiting function")
    return np.array(
        [
            acc_stats[0],
            test_ts_stats[0],
            train_ts_stats[0],
            acc_stats[1],
            test_ts_stats[1],
            train_ts_stats[1],
        ]
    )


def evaluate_test_models_Per_Person(Ntrials, mode, test_size, base_dir):

    #### NEED TO FINISH THIS FUNCTION... HC FETURES AND CNN ARE NOT SET UP AND NO VARIABLES TO
    #### STORE THINGS DEFINED ABOVE
    logger.info("Exit evaluate_test_models_Per_Person")

    ## create splits of train and test data
    # splitter = StratifiedShuffleSplit(n_splits=Ntrials, test_size=0.2)
    # pdb.set_trace()
    # train, test = list(splitter.split(X, y))[0]
    Nepochs = 100
    hidden_dim = 5

    def calc_statistics(val_list):
        av = sum(val_list) / len(val_list)
        var = sum([(val_list[kk] - av) ** 2 for kk in range(len(val_list))]) / len(
            val_list
        )
        return av, var

    ## load data
    X = np.load("../data/dopnet/X.npy", allow_pickle=True)
    y = np.load("../data/dopnet/y.npy")
    groups = np.load("../data/dopnet/groups.npy")
    max_time_dim = max([x.shape[1] for x in X])
    persons = sorted(list(set(groups)))

    logger.info("Starting Different Persons trials")
    logger.info("Persons: {}".format(persons))
    person_results = []
    for ii in range(len(persons)):
        logger.info("Person: {}".format(persons[ii]))
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
        #                                                     shuffle=True)
        idx = np.where(groups == persons[ii])[0]
        not_idx = np.where(groups != persons[ii])[0]
        X_train = [X[ii] for ii in not_idx]
        y_train = [y[ii] for ii in not_idx]
        X_test = [X[ii] for ii in idx]
        y_test = [y[ii] for ii in idx]
        # X_train = process_data(X_train, mode)
        # X_test = process_data(X_test, mode)
        logger.info("Group size for person {}: {}".format(ii, len(X_train)))
        logger.info(
            "Data and labels shape: {}; {}; {}; {}".format(
                len(X_train), len(y_train), len(X_test), len(y_test)
            )
        )
        logger.info("Data processed and going to start running trials")

        results = []
        # save_dir = os.path.join(base_dir, 'person_{}'.format(persons[ii]))
        logger.info("person {}".format(persons[ii]))
        for jj in range(Ntrials):
            logger.info("Trial {} out of {}".format(jj, Ntrials))
            name = "Person_{}_trial_{}".format(persons[ii], jj)
            logger.info("Ran Person/Trial: {}/{}".format(persons[ii], jj))
            results.append(
                run_all_trial(
                    X_train, X_test, y_train, y_test, max_time_dim, mode, base_dir, name
                )
            )
        logger.info("Trials over - results = {}".format(results))

        av_acc = []
        av_test_ts = []
        av_train_ts = []
        var_acc = []
        var_test_ts = []
        var_train_ts = []
        for ii in range(len(results[0])):
            res = np.array(results)
            # acc_stats = calc_statistics(results[ii][0])
            acc_stats = calc_statistics(res[:, ii, 0])
            logger.info("accuracy result_stats for trial {}: {}".format(ii, acc_stats))
            av_acc.append(acc_stats[0])
            var_acc.append(acc_stats[1])

            # train_ts_stats = calc_statistics(results[ii][1])
            train_ts_stats = calc_statistics(res[:, ii, 1])
            logger.info(
                "train time result_stats for trial {}: {}".format(ii, train_ts_stats)
            )
            av_train_ts.append(train_ts_stats[0])
            var_train_ts.append(train_ts_stats[1])

            # test_ts_stats = calc_statistics(results[ii][2])
            test_ts_stats = calc_statistics(res[:, ii, 2])
            logger.info(
                "test time result_stats for trial {}: {}".format(ii, test_ts_stats)
            )
            av_test_ts.append(test_ts_stats[0])
            var_test_ts.append(test_ts_stats[1])

        person_results.append(
            [av_acc, av_train_ts, av_test_ts, var_acc, var_train_ts, var_test_ts]
        )
        logger.info("Per Person Results: {}".format(person_results))
        logger.info("Exit evaluate_test_models_Per_Person")
    return person_results


def main():
    # SetLoggingLevel("INFO")
    SetLoggingLevel("DEBUG")

    experiment_selection = [False for _ in range(5)]

    # Prompt user for which experiment(s) to run
    while True:
        m = re.search(r"^([1-5])$", input("Which experiment to run? (1-5) ").strip())
        if m:
            experiment_index = int(m.groups(0)[0]) - 1
            experiment_selection[experiment_index] = True
            break

    params = {
        "feature": 3,  # whether to convert data to real
        "threshold_method": "yen",  # threshold method to use ("yen" or "otsu")
        "unitise": False,  # whether to convert data to binary (overrides normalise)
        "normalise": True,  # whether to normalise to [-1, 0]
        # "test_sizes": [0.2, 0.4, 0.6, 0.8, 0.9],    # test size ratios to use
        "test_sizes": [0.2],  # test size ratios to use
        "n_splits": 5,  # number of times to run each test size
        "methods": ["method1"],
    }

    ## EXPERIMENT:
    Ntrials = 1
    logger.info("-----------------------------------------------------")
    logger.info(f"Running experiment {experiment_index + 1}")

    ## Experiment 1: Evaluate models for different preprocessing steps and create latex tables
    test_size = 0.20
    if experiment_selection[0]:
        base_dir = os.path.join(BASE_DIR, "experiment1")
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        logger.info("Running Experiment 1 - results in {}".format(base_dir))
        logger.info("-----------------------------------------------------")
        logger.info("Run with binary preprocessing")
        bin_results = list(evaluate_test_models(Ntrials, "binary", test_size, base_dir))
        logger.info("Run with normalized preprocessing")
        norm_results = list(
            evaluate_test_models(Ntrials, "normalize", test_size, base_dir)
        )
        # bin_results = [
        #     av_acc_bin, av_train_ts_bin, av_test_ts_bin, var_acc_bin, var_train_ts_bin,
        #     var_test_ts_bin
        # ]
        # norm_results = [
        #     av_acc_norm, av_train_ts_norm, av_test_ts_norm, var_acc_norm, var_train_ts_norm,
        #     var_test_ts_norm
        # ]
        print_result_table(bin_results, norm_results, base_dir)
        logger.info("Experiment 1 complete")
        logger.info("-----------------------------------------------------")

    ## Experiment 2: Train on one person and test on the others for each preprocessing, generate
    ## plots
    if experiment_selection[1]:
        base_dir = os.path.join(BASE_DIR, "experiment2")
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        bin_results_person = np.array(
            evaluate_test_models_Per_Person(Ntrials, "binary", test_size, base_dir)
        )
        norm_results_person = np.array(
            evaluate_test_models_Per_Person(Ntrials, "normalize", test_size, base_dir)
        )

        nonpca_model_names = [
            "Gaussian SVD SVM",
            "Grassmann SVD SVM",
            "Laplace SVD SVM",
        ]
        pca_model_names = ["CNN1", "CNN2", "PCA KNN", "PCA LocSVM16", "PCA LocSVM64"]

        make_single_person_plots(
            norm_results_person[:, :, :3],
            bin_results_person[:, :, :3],
            "raw",
            nonpca_model_names,
            base_dir,
        )
        make_single_person_plots(
            norm_results_person[:, :, 3:],
            bin_results_person[:, :, 3:],
            "PCA",
            pca_model_names,
            base_dir,
        )

    ## Experiment 3: Evaluate models for a range of training set size, generate plots
    if experiment_selection[2]:
        base_dir = os.path.join(BASE_DIR, "experiment3")
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        test_sizes = [0.8, 0.6, 0.4, 0.2]
        nonpca_result_sizes = []
        pca_result_sizes = []
        for ii in range(len(test_sizes)):
            dir_path = os.path.join(
                base_dir, "test_size_{}%".format(int(test_sizes[ii] * 100))
            )
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            bin_results = evaluate_test_models(
                Ntrials, "binary", test_sizes[ii], dir_path
            )
            norm_results = evaluate_test_models(
                Ntrials, "normalize", test_sizes[ii], dir_path
            )
            nonpca_result_sizes.append([bin_results[:, :3], norm_results[:, :3]])
            pca_result_sizes.append([bin_results[:, 3:], norm_results[:, 3:]])

        nonpca_model_names = [
            "Gaussian SVD SVM",
            "Grassmann SVD SVM",
            "Laplace SVD SVM",
        ]
        pca_model_names = ["CNN1", "CNN2", "PCA KNN", "PCA LocSVM16", "PCA LocSVM64"]

        plot_class_size_results(
            nonpca_result_sizes, nonpca_model_names, "raw", base_dir
        )
        plot_class_size_results(pca_result_sizes, pca_model_names, "PCA", base_dir)

        print_result_table(
            nonpca_result_sizes[3][0],
            nonpca_result_sizes[3][1],
            nonpca_model_names,
            "raw",
            base_dir,
        )
        print_result_table(
            pca_result_sizes[3][0],
            pca_result_sizes[3][1],
            pca_model_names,
            "PCA",
            base_dir,
        )

    ## Experiment 4: Evaluate PCA models across dimension
    if experiment_selection[3]:
        base_dir = os.path.join(BASE_DIR, "experiment4")
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        logger.info("Loading Data")
        X = np.load("../data/dopnet/X.npy", allow_pickle=True)
        y = np.load("../data/dopnet/y.npy")
        groups = np.load("../data/dopnet/groups.npy")
        max_time_dim = max([x.shape[1] for x in X])

        modes = ["binary", "normalize"]
        N_trials = 1
        pca_Ns = range(1, 33)
        methods = ["PCA KNN", "LocSVM16", "LocSVM64"]
        method_names = [f"{method} ({mode})" for mode in modes for method in methods]

        results = np.zeros((N_trials, len(modes), len(pca_Ns), len(methods)))

        def load_or_save_test(path, test):
            if os.path.exists(path):
                logger.info("Loading from file...")
                with open(path, "rb") as f:
                    res = pickle.load(f)
            else:
                res = test()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(res, f)
            return res

        for n_trial in range(N_trials):
            X_train_, X_test_, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=True
            )
            for m, mode in enumerate(modes):
                if True:
                    X_train = process_data(X_train_, mode)
                    X_test = process_data(X_test_, mode)
                else:
                    X_train = X_train_
                    X_test = X_test_
                for pca_n in pca_Ns:
                    logger.info(f"n_trial = {n_trial + 1}/{N_trials}")
                    test_path = os.path.join(
                        base_dir, f"trial_{n_trial}", mode, f"pca_n_{pca_n}"
                    )

                    # PCA KNN
                    logger.info(
                        "Running PCA_KNN {} - hidden_dim = {}, K = {}".format(
                            mode, pca_n, 5
                        )
                    )
                    PCA_KNN_results = load_or_save_test(
                        os.path.join(test_path, "PCA_KNN.pkl"),
                        lambda: run_PCA_KNN_trial(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            max_time_dim,
                            "PCA_KNN",
                            pca_n=pca_n,
                        ),
                    )
                    logger.info(f"PCA KNN results: {PCA_KNN_results}")

                    # LocSVM16
                    n = 16
                    q = min(pca_n, 2)
                    logger.info(
                        f"Running LocSVM16 {mode} - n = {n}, q = {q}, pca_n = {pca_n}"
                    )
                    LocSVM16_results = load_or_save_test(
                        os.path.join(test_path, "LocSVM16.pkl"),
                        lambda: run_LocSVM_trial(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            max_time_dim,
                            n,
                            q,
                            "LocSVM16",
                            pca_n=pca_n,
                        ),
                    )
                    logger.info(f"LocSVM16 results: {LocSVM16_results}")

                    # LocSVM64
                    n = 64
                    q = min(pca_n, 18)
                    logger.info(
                        f"Running LocSVM64 {mode} - n = {n}, q = {q}, pca_n = {pca_n}"
                    )
                    LocSVM64_results = load_or_save_test(
                        os.path.join(test_path, "LocSVM64.pkl"),
                        lambda: run_LocSVM_trial(
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            max_time_dim,
                            n,
                            q,
                            "LocSVM64",
                            pca_n=pca_n,
                        ),
                    )
                    logger.info(f"LocSVM64 results: {LocSVM64_results}")

                    results[n_trial, m, pca_n - pca_Ns[0], 0] = PCA_KNN_results[0]
                    results[n_trial, m, pca_n - pca_Ns[0], 1] = LocSVM16_results[0]
                    results[n_trial, m, pca_n - pca_Ns[0], 2] = LocSVM64_results[0]

        with open(os.path.join(base_dir, "results.pkl"), "wb") as file:
            pickle.dump(results, file)

        results = np.swapaxes(results, 1, 2)
        results = results.reshape(*results.shape[:2], -1)
        var = np.var(results, axis=0)
        results = np.mean(results, axis=0)

        for i in range(results.shape[1]):
            plt.errorbar(pca_Ns, results[:, i], yerr=var[:, i])
        plt.legend(method_names)
        plt.xlabel("Dimension")
        plt.ylabel("Accuracy")
        plt.xticks(pca_Ns[::2])
        plt.yticks(np.linspace(0.4, 1, 7))
        plt.xlim(pca_Ns[0], pca_Ns[-1])
        plt.ylim(0.38, 1)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.grid(True, "major", axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "pca_acc_vs_dim.png"))
        plt.show()

    # Plot singular values
    if experiment_selection[4]:
        base_dir = os.path.join(BASE_DIR, "experiment5")
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        if False:
            logger.info("Loading Data")
            X = np.load("../data/dopnet/X.npy", allow_pickle=True)
            y = np.load("../data/dopnet/y.npy")
            groups = np.load("../data/dopnet/groups.npy")
            max_time_dim = max([x.shape[1] for x in X])

            for mode in ["binary", "normalize"]:
                X_ = process_data(X, mode)
                X_, data_dim = pad_data(X_, max_time_dim)
                vals = [[] for _ in range(4)]

                for i, x in enumerate(X_):
                    _, s, _ = np.linalg.svd(x)
                    vals[y[i]].append(s)
                    if True or i + 1 % 100 == 0 or i + 1 == len(X):
                        print(f"{mode} {i + 1}/{len(X)}", end="\r")
                print()

                with open(os.path.join(base_dir, f"svals_{mode}.pkl"), "wb") as f:
                    pickle.dump(vals, f)
        else:
            with open(os.path.join(base_dir, "svals_binary.pkl"), "rb") as f:
                vals_binary = pickle.load(f)
            with open(os.path.join(base_dir, "svals_normalize.pkl"), "rb") as f:
                vals_normalize = pickle.load(f)

            for i in range(len(vals_binary)):
                vals = np.array(vals_binary[i])
                plt.plot(np.arange(len(vals[0]))[:100], np.mean(vals, axis=0)[:100])
            # plt.ylim(10**(-4), 10**3)
            # plt.yscale('log')
            plt.xlabel("Index")
            plt.ylabel("Singular Value")
            plt.legend(
                ["Gesture: Wave", "Gesture: Pinch", "Gesture: Swipe", "Gesture: Click"]
            )
            plt.title("Binary Data Average Singular Values")
            plt.savefig(os.path.join(base_dir, "bin_data_singular_values.png"))
            plt.show()

            for i in range(len(vals_normalize)):
                vals = np.array(vals_normalize[i])
                plt.plot(np.arange(len(vals[0]))[:100], np.mean(vals, axis=0)[:100])
            # plt.ylim(10**(-4), 10**3)
            # plt.yscale('log')
            plt.xlabel("Index")
            plt.ylabel("Singular Value")
            plt.legend(
                ["Gesture: Wave", "Gesture: Pinch", "Gesture: Swipe", "Gesture: Click"]
            )
            plt.title("Normalized Data Average Singular Values")
            plt.savefig(os.path.join(base_dir, "norm_data_singular_values.png"))
            plt.show()

    print("Experiments complete")
    return None


if __name__ == "__main__":
    main()
