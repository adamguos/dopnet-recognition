"""
Examples demonstrating how to import and preprocess the DopNet dataset, and how to fit and evaluate the models.
"""

import numpy as np
import sklearn

from src import hermite_classifier, manifold_svm, preprocessing
from tests import data, dopnet, models


def import_and_preprocess_data():
    """
    Import DopNet data with spectrogram mean power features and preprocess the data.
    """
    # X:      spectrogram features
    # y:      class label (0-3)
    # groups: patient (A-F)
    # See table I in Ritchie and Jones for available features.
    X, y, groups = dopnet.import_all_data_with_groups(3)

    # Keep one of every 100 data points for the sake of quick demonstration
    X, y, groups = X[::100], y[::100], groups[::100]

    # Pad each spectrogram with zeros to be the same length
    X, _ = data.pad_data(X)
    X = np.array(X)

    # Preprocess Dopnet data by converting to a binary mask
    threshold = preprocessing.Threshold(unitise=True)
    X = threshold.transform(X)

    return X, y, groups


def main():
    # Import data
    X, y, groups = import_and_preprocess_data()

    # PCA-based methods (LocSVM and KNN) require flattening each 2D spectrogram into a 1D array and
    # then applying PCA
    pca = sklearn.decomposition.PCA(n_components=10)
    X_pca = pca.fit_transform(X.reshape(X.shape[0], -1))

    # Split into train and test sets
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    train_indices, test_indices = list(sss.split(X, y))[0]
    X_train, X_test, y_train, y_test, X_train_pca, X_test_pca = (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
        X_pca[train_indices],
        X_pca[test_indices],
    )

    # Use LocSVM (uses PCA data)
    #   n = degree of Hermite function
    #   q = dimension of data manifold
    hermite_svm = hermite_classifier.HermiteSVM(n=16, q=10)
    hermite_svm.fit(X_train_pca, y_train)
    y_predict = hermite_svm.predict(X_test_pca)
    print(f"HermiteSVM accuracy: {sum(y_test == y_predict) / len(y_test)}\n")

    # Use SVDSVM
    #   norm_sq = True corresponds to the Gaussian kernel, i.e. Gaussian SVM
    #   norm_sq = False corresponds to the Laplace kernel, i.e. Laplace SVM
    svd_svm = manifold_svm.SVDSVM(hidden_dim=10, norm_sq=False)
    svd_svm.fit(X_train, y_train)
    y_predict = svd_svm.predict(X_test)
    print(
        f"SVDSVM accuracy: {sum(y_test == y_predict) / len(y_test)} (norm_sq = {svd_svm.norm_sq})\n"
    )

    # Use GrassmannSVM
    grassmann_svm = manifold_svm.GrassmannSVM(hidden_dim=10)
    grassmann_svm.fit(X_train, y_train)
    y_predict = grassmann_svm.predict(X_test)
    print(f"GrassmannSVM accuracy: {sum(y_test == y_predict) / len(y_test)}\n")

    # Use KNN (uses PCA data)
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    y_predict = knn.predict(X_test_pca)
    print(f"KNN accuracy: {sum(y_test == y_predict) / len(y_test)}\n")

    # Use CNN
    print("Training CNN...")
    cnn = models.train_Model(X_train, y_train, 1, [1, 1, 1], [1, 1, 1])
    accuracy = models.test_Model(X_test, y_test, cnn)
    print(f"CNN accuracy: {accuracy}\n")


if __name__ == "__main__":
    main()
