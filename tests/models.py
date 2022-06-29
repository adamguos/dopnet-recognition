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
import scipy as sp
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

################################################################################
import logging
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

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('debug.log')
    logger.addHandler(fh)
#########################

# SetLoggingLevel("INFO")
SetLoggingLevel("DEBUG")


class DatasetSVD:
    def __init__(self, X, Y, regenerate=False):
        ## Saves parameters
        self.X=X
        self.Y=Y
        self.Xsvd=[]
        self.Ysvd=[]
        self.regenerate=regenerate

        ## regenerate data if new trial
        if self.regenerate:
            self._calc_svds()
        else:
            try:
                ## load file
                #os.load('not_exist','rb')
                pass
            except:
                self._calc_svds()   
        return

    def _calc_svds(self):
        Ny, Nx = len(self.Y), len(self.X)
        N = [Ny, Nx]
        logger.info("N: ".format(N))
        Imax = np.argmax(np.stack(N))
        for ii in range(N[Imax]):
            if ii < Nx:
                self.Xsvd.append(np.linalg.svd(self.X[ii], full_matrices=False))
            if ii < Ny:
                self.Ysvd.append(np.linalg.svd(self.Y[ii], full_matrices=False))
        return

    def save_svds(self, file_name):
        pass
        return

    def load_svds(self, file_name):
        pass
        return

    def get_svds(self):
        #pdb.set_trace()
        # TODO: reorg/complete this class and move this out of the function
        if (len(self.Xsvd) == 0) and (len(self.Ysvd) == 0):
            self._calc_svds()
        return self.Xsvd, self.Ysvd

class GrassmannSVM(svm.SVC):
    """
    SVM classifier using Gaussian kernel and projection metric on Grassmann manifold.
    Intended to not be used with ARMA parametrisation.
    """
    def __init__(self, kern_gamma=0.2, hidden_dim=10):
        self.kern_gamma = kern_gamma
        self.hidden_dim = hidden_dim

        # sklearn expects kern to be a function taking two arrays of data X and Y,
        # returning the value of kern(x, y) at each x \in X, y \in Y
        # should return an array of floats, shape (len(X), len(Y))
        def kern(X, Y):
            kern_matrix = np.zeros((len(X), len(Y)))
            Xsvd = [ sp.linalg.svd(x, full_matrices=False) for x in X]
            Ysvd = [ sp.linalg.svd(y, full_matrices=False) for y in Y]
            # Xsvd_.append(sp.linalg.svd(X, full_matrices=False))
            # Xsvd, Ysvd = svds.get_svds()
            # logger.info('Xsvd length: {}'.format(len(Xsvd)))
            # logger.info('Ysvd length: {}'.format(len(Ysvd)))
            from tqdm import tqdm
            for ii in tqdm(range(len(X))):
                Xu = Xsvd[ii][0][:, :hidden_dim]
                r = Xu.shape[1]
                for jj in range(len(Y)):
                    Yu = Ysvd[jj][0][:, :hidden_dim]
                    #logger.info('Xu and Yu shape:  {} and {}'.format(Xu.shape,Yu.shape))
                    kern_matrix[ii, jj] = np.exp(-kern_gamma*(r - np.linalg.norm(Xu.T @ Yu) ** 2))
                    #print(kern_matrix[ii,jj])
            return kern_matrix
        super().__init__(kernel=kern)
        return

    def transform(self, X):
        return super().predict(X)

class RManSVM(svm.SVC):
    """
    SVM classifier using Gaussian kernel and projection metric on Grassmann manifold.
    Intended to not be used with ARMA parametrisation.
    """
    def __init__(self, kern_gamma=0.2, hidden_dim=10):
        self.kern_gamma = kern_gamma
        self.hidden_dim = hidden_dim

        # sklearn expects kern to be a function taking two arrays of data X and Y,
        # returning the value of kern(x, y) at each x \in X, y \in Y
        # should return an array of floats, shape (len(X), len(Y))
        #def kern(X, Y):
            # kern_matrix = np.zeros((len(X), len(Y)))
            # # svds = DatasetSVD(X, Y)
            # # Xsvd, Ysvd = svds.get_svds()
            # for ii in range(len(X)):
            #     Xsvd = np.linalg.svd(X[ii], full_matrices=False)
            #     Xr = Xsvd[0][:, :hidden_dim] @ Xsvd[2][:hidden_dim, :]  
            #     for jj in range(len(Y)):
            #         Ysvd = np.linalg.svd(X[ii], full_matrices=False)
            #         Yr = Ysvd[0][:, :hidden_dim] @ Ysvd[2][:hidden_dim, :]  
            #         logger.info('Xr and Yr shape:  {} and {}'.format(Xr.shape,Yr.shape))
            #         kern_matrix[ii, jj] = np.exp(-kern_gamma*np.linalg.norm(Xr.T @ Yr, 2) ** 2)
            #         logger.info('Xr and Yr shape:  {} and {}'.format(Xr.shape,Yr.shape))
            # return kern_matrix
            #logger.info('Kernel matrix: {}'.format(kern_matrix) )
        #super().__init__(kernel=kern)
        super().__init__(kernel=self.gpu_compute_kern)

        return

    # def gpu_compute_kern(self, X, Y):
    #     import cupy as cp
    #     from tqdm import tqdm
    #     kern_matrix = cp.zeros((len(X), len(Y)))
    #     start_trial = time.time()
    #     # X = [cp.asarray(X[ii]) for ii in range(len(X))]
    #     Y = [cp.asarray(Y[ii]) for ii in range(len(Y))]
    #     for ii in tqdm(range(len(X))):
    #     #for ii in range(len(X)):
    #         #pdb.set_trace()
    #         tic = time.time()
    #         Xsvd = cp.linalg.svd(cp.asarray(X[ii]), full_matrices=False)
    #         toc = time.time() - tic
    #         print("svd time: ", toc)
    #         tic = time.time()
    #         Xr = Xsvd[0][:, :self.hidden_dim] @ Xsvd[2][:self.hidden_dim, :]  
    #         toc = time.time()- tic
    #         print("mat mult time ", toc)
    #         for jj in range(len(Y)):
    #             #pdb.set_trace()
    #             Ysvd = cp.linalg.svd(Y[ii], full_matrices=False)
    #             Yr = Ysvd[0][:, :self.hidden_dim] @ Ysvd[2][:self.hidden_dim, :]
    #             #pdb.set_trace()  
    #             #logger.info('Xr and Yr shape:  {} and {}'.format(Xr.shape,Yr.shape))
    #             kern_matrix[ii, jj] = cp.exp(-self.kern_gamma*cp.linalg.norm(Xr.T @ Yr, 2) ** 2)
    #             #logger.info('Xr and Yr shape:  {} and {}'.format(Xr.shape,Yr.shape))
    #         logger.info('Outer iteration time {}: {}'.format(ii, time.time() - start_trial))
    #     kern = np.asarray(kern_matrix)
    #     return kern

    def gpu_compute_kern(self, X, Y):
        #import cupy as cp
        from tqdm import tqdm
        kern_matrix = np.zeros((len(X), len(Y)))
        Xsvd = [ sp.linalg.svd(x, full_matrices=False) for x in X]
        Ysvd = [ sp.linalg.svd(y, full_matrices=False) for y in Y]
        for ii in range(len(Xsvd)):
            Xr = Xsvd[ii][0][:, :self.hidden_dim] @ Xsvd[ii][2][:self.hidden_dim, :]  
            for jj in range(len(Y)):
                Yr = Ysvd[jj][0][:, :self.hidden_dim] @ Ysvd[jj][2][:self.hidden_dim, :]
                kern_matrix[ii, jj] = np.exp(-self.kern_gamma*np.linalg.norm(Xr.T @ Yr, 2) ** 2)
        kern = np.asarray(kern_matrix)
        return kern


    def transform(self, X):
        return super().predict(X)        

class ProdGrassmannSVM(svm.SVC):
    """
    SVM classifier using Gaussian kernel and projection metric on Grassmann manifold.
    Intended to not be used with ARMA parametrisation. cp.array([1, 2, 3])
    """
    def __init__(self, kern_gamma=0.2, hidden_dim=10):
        self.kern_gamma = kern_gamma
        self.hidden_dim = hidden_dim

        # sklearn expects kern to be a function taking two arrays of data X and Y,
        # returning the value of kern(x, y) at each x \in X, y \in Y
        # should return an array of floats, shape (len(X), len(Y))
        def kern(X, Y):
            kern_matrix = np.zeros((len(X), len(Y)))
            Xsvd = [ sp.linalg.svd(x, full_matrices=False) for x in X]
            Ysvd = [ sp.linalg.svd(y, full_matrices=False) for y in Y]
            # logger.info('Xsvd length: {}'.format(len(Xsvd)))
            # logger.info('Ysvd length: {}'.format(len(Ysvd)))
            for ii in range(len(X)):
                Xu = Xsvd[ii][0][:, :hidden_dim]
                XvT = Xsvd[ii][2][:, :hidden_dim]
                rx = hidden_dim
                ry = hidden_dim
                for jj in range(len(Y)):
                    Yu = Ysvd[jj][0][:, :hidden_dim]
                    YvT = Ysvd[jj][2][:, :hidden_dim]
                    #logger.info('Xu and Yu shape:  {} and {}'.format(Xu.shape,Yu.shape))
                    kern_matrix[ii, jj] = np.exp(-kern_gamma*((rx - np.linalg.norm(Xu.T @ Yu) + (rx - np.linalg.norm(XvT.T @ YvT)) ** 2)))
            return kern_matrix
        super().__init__(kernel=kern)
        return

    def transform(self, X):
        return super().predict(X)


def frobenius_kernel(X, Y, alpha, beta, gamma, hidden_dim, include_v=False, norm_sq=False):
    """
    Computes a kernel using the Frobenius norm on the singular value decompositions of multi-feature
    signals in Y1 and Y2.
    """
    # Compute kern(x, y) for all x \in X, y \in Y
    # Store into array of size (len(X), len(Y))
    kern_matrix = np.empty((len(X), len(Y)))
    kern_matrix[:] = np.nan
    # svds = DatasetSVD(X, Y)
    # Xsvd, Ysvd = svds.get_svds()
    Xsvd = [ sp.linalg.svd(x, full_matrices=False) for x in X]
    Ysvd = [ sp.linalg.svd(y, full_matrices=False) for y in Y]
    for ii in range(len(X)):
        U1 = Xsvd[ii][0][:, :hidden_dim]
        S1 = Xsvd[ii][1][:hidden_dim]
        Vh1 = Xsvd[ii][2][:hidden_dim, :]
        for jj in range(len(Y)):
            U2 = Ysvd[jj][0][:, :hidden_dim]
            S2 = Ysvd[jj][1][:hidden_dim]
            V2h = Ysvd[jj][2][:hidden_dim, :]
            if norm_sq:     # Gaussian kernel
                kern_matrix[ii, jj] = -beta * (np.linalg.norm(S1 - S2) ** 2) \
                        - alpha * (np.linalg.norm(U1 - U2) ** 2)
            else:           # Laplace kernel
                kern_matrix[ii, jj] = -beta * (np.linalg.norm(S1 - S2)) \
                        - alpha * (np.linalg.norm(U1 - U2))

            if include_v:   # include right singular vectors if requested (generally reduces performance)
                kern_matrix[ii, jj] -= gamma * np.linalg.norm(V1[:V2.shape[0], :V2.shape[1]] \
                        - V2[:V1.shape[0], :V1.shape[1]])
    kern_matrix = np.exp(kern_matrix)
    return kern_matrix

class SVDSVM(svm.SVC):
    """
    SVM classifier using Frobenius norm on singular value decompositions between multi-feature signals.
    For use in sklearn.pipeline.
    """
    def __init__(self, a=0.1, b=0.1, c=None, hidden_dim=10, scaled=False, include_v=False, norm_sq=False, max_iter=-1):
        self.a = a
        self.b = b
        self.c = a if c is None else c
        self.hidden_dim = hidden_dim
        self.scaled = scaled
        self.include_v = include_v
        self.norm_sq = norm_sq

        kern = lambda Y1, Y2: frobenius_kernel(Y1, Y2, self.a, self.b, self.c, hidden_dim=self.hidden_dim,
                include_v=self.include_v, norm_sq=self.norm_sq)

        super().__init__(kernel=kern,max_iter=max_iter)

    def transform(self, X):
        return super().predict(X)

class SVDSVM(svm.SVC):
    """
    SVM classifier using Frobenius norm on singular value decompositions between multi-feature signals.
    For use in sklearn.pipeline.
    """
    def __init__(self, a=0.1, b=0.1, c=None, hidden_dim=10, scaled=False, include_v=False, norm_sq=False, max_iter=-1):
        self.a = a
        self.b = b
        self.c = a if c is None else c
        self.hidden_dim = hidden_dim
        self.scaled = scaled
        self.include_v = include_v
        self.norm_sq = norm_sq

        kern = lambda Y1, Y2: frobenius_kernel(Y1, Y2, self.a, self.b, self.c, hidden_dim=self.hidden_dim,
                include_v=self.include_v, norm_sq=self.norm_sq)

        super().__init__(kernel=kern,max_iter=max_iter)

    def transform(self, X):
        return super().predict(X)

# Nfilts = [8,16,32]
# Nsize = [20,10,5]
class conv3Network(torch.nn.Module):
    def __init__(self,Nfilts,Nsize,Nclasses,sig_size):
        super(conv3Network,self).__init__()
        
        self.W1 = nn.Conv2d(1,Nfilts[0],Nsize[0])               # Nfilts x input_size - Nsize[1] + 1
        self.W2 = nn.Conv2d(Nfilts[0],Nfilts[1],Nsize[1])       # Nfilts**2 x (input_size - Nsize[2] - Nsize[1] + 2)/2
        self.W3 = nn.Conv2d(Nfilts[1],Nfilts[2],Nsize[2])       # Nfilts**3 x (input_size - Nsize[2] - Nsize[1] + 2)/4 - Nsize[3]/2 + 1
        output_size = self.conv_output_size(sig_size, Nsize)
        self.f1 = nn.Linear(int(Nfilts[2]*output_size[0]*output_size[1]),Nclasses)
        self.pool = nn.MaxPool2d(2,2)

        # self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
        return None        

    def forward(self, x):
        h1 = self.pool(self.relu(self.W1(x)))
        h2 = self.pool(self.relu(self.W2(h1)))
        h3 = self.pool(self.relu(self.W3(h2)))
        hout = self.f1(h3.view(-1, np.prod(h3.shape[1:])))
        return hout

    def conv_output_size(self, sig_size, Nsize, num_layers=None):
        sig_size0 = sig_size[0]
        sig_size1 = sig_size[1]
        if num_layers is None:
            for ii in range(len(Nsize)):
                sig_size0 = (sig_size0-Nsize[ii]+1)//2
                sig_size1 = (sig_size1-Nsize[ii]+1)//2
                #print(sig_size0)
                #print(sig_size1)
        return (sig_size0, sig_size1)

class conv7Network(torch.nn.Module):
    def __init__(self,Nfilts,Nsize,Nclasses,sig_size):
        super(conv7Network,self).__init__()
    
        self.W1 = nn.Conv2d(1,Nfilts[0],Nsize[0])               # Nfilts x input_size - Nsize[1] + 1
        self.W2 = nn.Conv2d(Nfilts[0],Nfilts[1],Nsize[1])       # Nfilts**2 x (input_size - Nsize[2] - Nsize[1] + 2)/2
        self.W3 = nn.Conv2d(Nfilts[1],Nfilts[2],Nsize[2])       # Nfilts**3 x (input_size - Nsize[2] - Nsize[1] + 2)/4 - Nsize[3]/2 + 1
        self.W4 = nn.Conv2d(Nfilts[2],Nfilts[3],Nsize[3])
        self.W5 = nn.Conv2d(Nfilts[3],Nfilts[4],Nsize[4])
        self.W6 = nn.Conv2d(Nfilts[4],Nfilts[5],Nsize[5])
        self.W7 = nn.Conv2d(Nfilts[5],Nfilts[6],Nsize[6])
        self.f1 = nn.Linear(int(Nfilts[6]*8*8),256)
        self.f2 = nn.Linear(256,Nclasses)
        self.pool = nn.AvgPool2d(2,2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(8)

        # self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(0.01)
        return None        

    def forward(self, x):

        h1 = self.pool(self.relu(self.W1(x)))
        h2 = self.pool(self.relu(self.W2(h1)))
        h3 = self.pool(self.relu(self.W3(h2)))
        h4 = self.pool(self.relu(self.W4(h3)))
        h5 = self.relu(self.W5(h4))
        h6 = self.relu(self.W6(h5))
        h7 = self.relu(self.adaptive_pool(self.W7(h6)))
        h8 = self.relu(self.f1(h7.view(-1, np.prod(h7.shape[1:]))))
        hout = self.f2(h8)
        return hout

    def conv_output_size(self, sig_size, Nsize, num_layers=None):
        sig_size0 = sig_size[0]
        sig_size1 = sig_size[1]
        if num_layers is None:
            for ii in range(len(Nsize)):
                sig_size0 = (sig_size0-Nsize[ii]+1)//2
                sig_size1 = (sig_size1-Nsize[ii]+1)//2
                #print(sig_size0)
                #print(sig_size1)
        return (sig_size0, sig_size1)

def train_Model(train_data, train_labels, Nepochs, Nfilts, Nsize):
    
    #pdb.set_trace()
    mbs = 16
    Ntrain = len(train_data)
    #Ntest = len(test_data)
    print('Ntrain ', Ntrain)
    #print('Ntest', Ntest)

    training_pairs = []
    # for ii in range(train_data.shape[0]):
    for ii in range(len(train_data)):
        training_pairs.append([np.expand_dims(train_data[ii],0), train_labels[ii]])
    trainloader = torch.utils.data.DataLoader(training_pairs, batch_size=mbs, shuffle=True, num_workers=8)
    sig_size = train_data[0].shape

    train_loss = []
    train_acc = []
    criterion = torch.nn.CrossEntropyLoss()
    Nclasses = 4
    #input_size = data_dim
    if len(Nfilts) == 3:
        model = conv3Network(Nfilts,Nsize,Nclasses,sig_size)
    elif len(Nfilts) == 7:
        model = conv7Network(Nfilts,Nsize,Nclasses,sig_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
    for ee in range(Nepochs):
        total_train_loss = 0
        total_train_acc = 0
        for data, label in trainloader:

            data, label =  data.to(device), label.to(device)  
            optimizer.zero_grad()
            model.train()
            pred = model(data.float())
            loss = criterion(pred,label.long())

            total_train_loss += loss.item() / Ntrain
            total_train_acc += (torch.argmax(pred,1) == label).float().sum().item() / Ntrain
            loss.backward()
            optimizer.step()
        
        #print('--- Training ---')
        # print(total_train_loss)
        # print(total_train_acc)

    train_loss.append(total_train_loss)
    train_acc.append(total_train_acc)
    return model

def test_Model(test_data,test_labels,model):

    mbs = 16
    Ntest = len(test_data)

    testing_pairs = []
    for ii in range(len(test_data)):
        testing_pairs.append([np.expand_dims(test_data[ii],0), test_labels[ii]])    
    testloader = torch.utils.data.DataLoader(testing_pairs, batch_size=mbs, shuffle=True,
            num_workers=12)

    ## evaluate on test set
    total_test_loss = 0
    total_test_acc = 0
    model.to(device)
    for data, label in testloader:
        data, label =  data.to(device), label.to(device)  
        pred = model(data.float())
        total_test_acc += (torch.argmax(pred,1) == label).float().sum().item() / Ntest

    #print('--- Testing ---')
    #print(total_test_loss)
    #print(total_test_acc)
    return total_test_acc

    








