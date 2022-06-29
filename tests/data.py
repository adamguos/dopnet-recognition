"""Contains test functions for DopNET dataset"""

import os
import pdb
import pickle
import sys
import time
from datetime import datetime
from types import SimpleNamespace

sys.path.append("../")
sys.path.append("../src")

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from skimage.filters import try_all_threshold, threshold_otsu, threshold_yen
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, LeaveOneGroupOut, train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from torch.autograd import Variable
import torch
import torch.nn as nn

#from manifold_svm import SVDSVM, GrassmannSVM, JointGrassmannSVM, find_best_hidden_dim, PSDSVM, KGSVM, kernel_testing
#from preprocessing import Threshold
from mpl_toolkits import mplot3d


#device = torch.device("cuda:0")
device = torch.device("cpu")


def export_all_data_with_groups():
    """Exports all data of dopnet dataset with info on groups for cross-validation"""
    signals = None
    labels = None
    groups = None

    for person in ["A", "B", "C", "D", "E", "F"]:
        test = scipy.io.loadmat(
            "/home/emason/hrushikesh_dopnet/Dop_Net_DATA/Data/Data_Per_PersonData_Training_Person_" + person + ".mat")

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

    pickle.dump(signals, open("../Dop_Net_DATA/Data/X.npy", "wb"))
    np.save("../Dop_Net_DATA/Data/y.npy", labels)
    np.save("../Dop_Net_DATA/Data/groups.npy", groups)

    print("exported data")

def import_all_data_with_groups(feature):
    """
    `feature` refers to Table I in Ritchie and Jones
        1: spectrogram summed intensity
        2: spectrogram variance
        3: spectrogram mean power (dB)
        4: spectrogram SVD -- summed singular values
        5: entropy of spectrogram intensity
    """
    X, y, groups = np.load("../Dop_Net_Data/Data/X.npy", allow_pickle=True), \
                   np.load("../Dop_Net_Data/Data/y.npy"), \
                   np.load("../Dop_Net_Data/Data/groups.npy")

    if feature == 1:
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
        raise ValueError("4: spectrogram SVD -- summed singular values not implemented yet")
    elif feature == 5:
        raise ValueError("5: entropy of spectrogram intensity not implemented yet")
    else:
        raise ValueError("Invalid feature")

    print(f"{datetime.now().strftime('%H:%M:%S')}: imported data\n")
    return X, y, groups


def envelope_Detection(data,thresh):

    M, N = data.shape
    thresh = thresh * .9
    up_data = data[:(M//2),:] * 1
    down_data = data[(M//2):,:] * 1
    out_data = data.copy()
    #data = normalize(data)
    #pdb.set_trace()
    up_inds = []
    down_inds = []
    for ii in range(N):
        idx = np.where( data[:,ii] >= thresh )[0]
        if idx.size == 0:
            up_inds.append(0)
            down_inds.append(M//2)
            out_data[:,ii] = 0
        else:
            min_idx = np.amin(idx)
            max_idx = np.amax(idx)
            up_inds.append(min_idx)
            down_inds.append(max_idx)
            out_data[:min_idx,ii] = 0
            out_data[max_idx:,ii] = 0

    return out_data

def calc_empirical_features(X=None,y=None):

    if X is None:
        X, y, groups = np.load("../Dop_Net_DATA/Data/dopnet/X.npy", allow_pickle=True), \
                       np.load("../Dop_Net_DATA/Data/y.npy"), \
                       np.load("../Dop_Net_DATA/Data/groups.npy")
    
    # threshold and envelope
    X_ = []
    for x in X:
        x = 20*np.log10(np.abs(x))
        thresh = threshold_yen(x)
        X_.append(envelope_Detection(x,thresh))
    X = X_

    ## extract features
    features = []
    for x in X:
        #pdb.set_trace()
        time_power = np.linalg.norm(x,axis=0)**2 /  np.max(np.linalg.norm(x,axis=0)**2)
        #pdb.set_trace()
        min_time = np.where(time_power >= 0.20)[0][0]
        max_time = np.where(time_power >= 0.20)[0][-1]
        freq_power = np.linalg.norm(x,axis=1)**2 /  np.max(np.linalg.norm(x,axis=1)**2)
        #pdb.set_trace()
        min_freq = np.where(freq_power >= 0.20)[0][0]
        max_freq = np.where(freq_power >= 0.20)[0][-1]
        #pdb.set_trace()
        T = (max_time - min_time)# / x.shape[1]    
        #pdb.set_trace()
        B =  (max_freq-min_freq)# / x.shape[0]
        #pdb.set_trace()
        R = np.abs( (max_freq - x.shape[0]/2 ) / (x.shape[0]/2 - min_freq) )# / x.shape[0]/2
        #pdb.set_trace()
        features.append([T, B, R])

    ## plot the class features
    features = np.asarray(features)
    #scaler = StandardScaler().fit(features)
    #features = scaler.transform(features)
    normalizer = MinMaxScaler().fit(features)
    features = normalizer.transform(features)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(features[:, 0], features[:, 1], features[:, 2], c = y)
    # plt.show()

    return features, y

def load_data(mode):

    X, y, groups = np.load("../Dop_Net_DATA/Data/X.npy", allow_pickle=True), \
                   np.load("../Dop_Net_DATA/Data/y.npy"), \
                   np.load("../Dop_Net_DATA/Data/groups.npy")

    # threshold and envelope
    X_ = []
    for x in X:
        x = 20 * np.log10(abs(x) / np.amax(abs(x)))
        thresh = threshold_yen(x)
        x[ x < thresh ] = 0
        x = np.abs(x)
        if mode == 'binary':
            x[x == 0] = 1
        elif mode == 'normalize':
            x = x / np.amax(np.abs(x))
        #X_.append(envelope_Detection(x,thresh))
        X_.append(x)
    return X_, y


def pad_data(X,max_len=None):
    if max_len is None:
        max_len = max([x.shape[1] for x in X])
    if max_len % 2 != 0:
        max_len += 1 
    #print('max_len: ',max_len)
    data_dim = (X[0].shape[0],max_len)        
    data_out = []
    for datum in X:
        #pdb.set_trace()
        len_diff = max_len-datum.shape[1]
        #print('len_diff: ',len_diff)
        if 1 < len_diff:
            padding = np.zeros((datum.shape[0],len_diff//2))        
            try:
                # if len_diff % 2 != 0:
                #     pdb.set_trace()
                padded_datum = np.concatenate((padding,np.concatenate((datum,padding),axis=1)),axis=1)
                # if len_diff % 2 != 0:
                #     pdb.set_trace()
                #print('data shape: ',padded_datum.shape[1])
                if padded_datum.shape[1] < max_len:
                    # pdb.set_trace()
                    padded_datum = np.concatenate((padded_datum,np.zeros((padded_datum.shape[0],1))),axis=1)
            except ValueError:
                print(ValueError)
        elif len_diff == 1:
            padded_datum = np.concatenate((datum,np.zeros((datum.shape[0],1))),axis=1)
        elif len_diff == 0:
            padded_datum = datum
        else:
            pdb.set_trace()
            print('wtf')
        data_out.append(padded_datum)
        #print(padded_datum.shape)
        #pdb.set_trace()
    return data_out, data_dim



# def process_data(X,mode):
#     # # threshold and envelope
#     # from sklearn.preprocessing import StandardScaler
#     X_ = []
#     for x in X:
#         x = 20 * np.log10((np.abs(x)))
#         x -= np.min(x) * np.ones_like(x)
#         x = x / np.amax(x)
#     return x
#     #     x = 20 * np.log10((abs(x) - np.amin(abs(x))*np.ones_like(np.abs(x))) / (np.amax(abs(x)) - np.amin(abs(x)))*np.ones_like(abs(x)))
#     #     pdb.set_trace()
#     #     # x = x / np.amax(x)
#     #     #thresh = threshold_yen(np.abs(x))
#     #     #x[ x < thresh ] = 0
#     #     #x = np.abs(x)
#     #     pdb.set_trace()
#     #     if mode == 'binary':
#     #         mag = np.abs(x)
#     #         mag[mag == 0] = 1
#     #         x = x / mag
#     #     elif mode == 'normalize':
#     #         x = x / np.amax(np.abs(x))
#     #     #X_.append(envelope_Detection(x,thresh))
#     #     X_.append(x)
#     #     pdb.set_trace()
#     # return X_

# def process_data(X,mode):
#     # threshold and envelope
#     X_ = []
#     for x in X:
#         x = 20*(np.log10((np.abs(x) / np.amax(abs(x)))))
#         x = (x-np.min(np.min(x))) / (np.max(np.max(x))-np.min(np.min(x)))
#         thresh = threshold_yen(x)
#         x[ x < thresh ] = 0
#         if mode == 'binary':
#             x[ x != 0] = 1
#         elif mode == 'normalize':
#             pass
#         #X_.append(envelope_Detection(x,thresh))
#         X_.append(x)
#     return X_


## OG
# def process_data(X,mode):
#     # threshold and envelope
#     X_ = []
#     for x in X:
#         x = 20 * np.log10(abs(x) / np.amax(abs(x)))
#         thresh = threshold_yen(x)
#         x[ x < thresh ] = 0
#         #x = np.abs(x)
#         if mode == 'binary':
#             mag = np.abs(x)
#             mag[mag == 0] = 1
#             x = x / mag
#         elif mode == 'normalize':
#             x = x / np.amax(np.abs(x))
#         #X_.append(envelope_Detection(x,thresh))
#         X_.append(x)
#     return X_

def process_data(X, mode):
    # threshold and envelope
    import scipy as sp
    X_ = []
    cnt = 0
    for x in X:
        x = 20 * np.log10(abs(x) / np.amax(abs(x)))
        thresh = threshold_yen(x)
        x[ x < thresh ] = 0
        #x = np.abs(x)
        if mode == 'binary':
            mag = np.abs(x)
            mag[mag == 0] = 1
            x = x / mag
        elif mode == 'normalize':
            x = x / np.amax(np.abs(x))
        #Xsvd_.append(sp.linalg.svd(x, full_matrices=False))
        X_.append(x)
        # print(cnt)
        cnt+=1
    return X_
