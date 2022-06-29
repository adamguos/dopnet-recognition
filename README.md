# A manifold learning approach for gesture recognition from micro-Doppler radar measurements

E. S. Mason, H. N. Mhaskar, A. Guo

[doi.org/10.1016/j.neunet.2022.04.024](https://doi.org/10.1016/j.neunet.2022.04.024)

## Data and setup

We use the [DopNet](https://dop-net.com/) radar dataset, which can be found on their [GitHub
repository](https://github.com/UCLRadarGroup/DopNet/blob/master/data/Data_Download_And_Details.md).

1. Download data from persons A to F (`Data_Per_PersonData_Training_Person_{A-F}.mat`) and put those
   six files into the `data/dopnet/` directory
2. Install required Python libraries (e.g. in a Conda environment): `pip install -r
   requirements.txt`
3. Install PyTorch
4. Generate `.npy` files: `python export_data.py`
5. The `example_usage.py` script demonstrates how to import and preprocess the dataset as well as
   run the models. The script itself can be run: `python example_usage.py`

## Experiments

To run the experiments presented in the paper: `cd tests` and `python experiments.py`. The following
five experiments are available:

1. Unused (replaced by experiment 3)
2. Evaluate each model by training on 5 subjects' data and testing on the last subject's data
3. Evaluate each model on different train/test split ratios
4. Evaluate PCA-based models across PCA dimensions
5. Plot singular values of spectrograms

The models used are referred to by the following names, corresponding to Tables 3 and 4 of the
paper:

- SVD-based models: Gaussian SVD SVM, Grassmann SVD SVM, Laplace SVD SVM
- PCA-based models: PCA KNN, PCA LocSVM16, PCA LocSVM64
- CNN models: CNN1, CNN2

For reference, the experiments were run on Arch Linux (kernel 5.x) using an AMD Ryzen 5 5600X with
32GB of memory. An NVIDIA GTX 1060 6GB was used to accelerate PyTorch for CNN training and
inference. The experiments may need modifications to run on less than 32GB of memory or without an
NVIDIA GPU.

## File rundown

- `src/manifold_svm.py`
    - Contains all the classification code. Classifiers inherit from `sklearn.SVM.SVC`, and only
      change the kernel function
- `src/preprocessing.py`
    - Contains a number of preprocessing classes designed to be used in `sklearn.Pipeline`
    - Only `Threshold` is used, which implements scaling to binary/[-1, 0] interval and Otsu/Yen
      thresholding
- `tests/export_data.py`
    - Run this to read and export DopNet data
    - Put all the `Data_Per_PersonData....mat` files into `data/dopnet/`
- `tests/dopnet.py`
    - Contains functions for running tests
    - As currently set up, run this file to execute 25 trials across 5 split sizes of Laplace
      kernel, normalised data
    - `test_split_sizes` is what we interface with to change the test being run
    - Other functions are there to help plot graphs for the report, provide wrapper functions for
      running/saving tests, etc.
- `tests/logs/read_tests.py`
    - Contains functions for quickly reading from the log files
    - Run `read_tests.py` to get an overview of test results in chronological order
