#!/usr/bin/env python3
"""
This is a module that implments a gzip classifier. You can test it by running the following command:
python -m gzip_classifier --compressor gzip --k 3 --m 100 --method random --distance_matrix None --dataset 20newsgroups
"""
# These lines will be used  to setup a virtual environment inside the current working directory in a folder called env
# You might need to install venv with:
# sudo apt-get install python3-venv
# python3 -m pip install venv
# python3 -m venv env
# source env/bin/activate
# run `deactivate` to exit the virtual environment
# These lines will be used to install the dependencies needed for this file
# You might need to install pip with:
# sudo apt-get install python3-pip
# python -m pip install numpy scikit-learn tqdm scikit-learn-extra pandas imbalanced-learn

import numpy as np
import warnings
import gzip
from tqdm import tqdm
from pathlib import Path
import logging
import time
import argparse
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from Levenshtein import distance, ratio, hamming, jaro, jaro_winkler, seqratio
import pandas as pd
from multiprocessing import cpu_count
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from joblib import Parallel, delayed
from typing import Literal

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def _gzip_len(x):
    return len(gzip.compress(str(x).encode()))


def _lzma_len(x):
    import lzma

    return len(lzma.compress(str(x).encode()))


def _bz2_len(x):
    import bz2

    return len(bz2.compress(str(x).encode()))


def _zstd_len(x):
    import zstd

    return len(zstd.compress(str(x).encode()))


def _pickle_len(x):
    import pickle

    return len(pickle.dumps(x))


def _brotli_len(x):
    import brotli

    return len(brotli.compress(str(x).encode()))


compressors = {
    "gzip": _gzip_len,
    "lzma": _lzma_len,
    "bz2": _bz2_len,
    "zstd": _zstd_len,
    "pkl": _pickle_len,
    "brotli": _brotli_len,
}

# Because these metrics describe similarity, we need to subtract them from 1 to get a distance
def ratio_distance(x1, x2):
    return 1 - ratio(x1, x2)

def jaro_distance(x1, x2):
    return 1 - jaro(x1, x2)

def jaro_winkler_distance(x1, x2):
    return 1 - jaro_winkler(x1, x2)

def seqratio_distance(x1, x2):
    return 1 - seqratio(x1, x2)
######################################################

string_metrics = {
    "levenshtein": distance,
    "hamming": hamming,
    "jaro": jaro_distance,
    "ratio" : ratio_distance,
    "jaro_winkler": jaro_winkler_distance,
    "seqratio": seqratio_distance,
}

all_metrics = {
    **compressors,
    **string_metrics,
}

all_condensers = [
    "sum",
    "mean",
    "medoid",
    "random",
    "knn",
    "svc",
    "hardness",
    "nearmiss",
    None,
    "None",
    "null",
    "",
]


transform_dict = {
    "abs": np.abs,
    "square": np.square,
    "exp": np.exp,
    "distance_rbf" : lambda x: 2 - 2 * np.exp(-x ** 2),
    "distance_rbf_gamma_001" : lambda x: 2 - 2 * np.exp(-x ** 2 / 0.001),
    "distance_rbf_gamma_01" : lambda x: 2 - 2 * np.exp(-x ** 2 / 0.01),
    "distance_rbf_gamma_1" : lambda x: 2 - 2 * np.exp(-x ** 2 / 0.1),
    "distance_rbf_gamma10" : lambda x: 2 - 2 * np.exp(-x ** 2 / 10),
    "distance_rbf_gamma100" : lambda x: 2 - 2 * np.exp(-x ** 2 / 100),
    "distance_rbf_gamma1000" : lambda x: 2 - 2 * np.exp(-x ** 2 / 1000),
    "exp_neg": lambda x: np.exp(-x),
    "exp_neg_gamma_001": lambda x: np.exp(-x / 0.001),
    "exp_neg_gamma_01": lambda x: np.exp(-x / 0.01),
    "exp_neg_gamma_1": lambda x: np.exp(-x / 0.1),
    "exp_neg_gamma10": lambda x: np.exp(-x / 10),
    "exp_neg_gamma100": lambda x: np.exp(-x / 100),
    "exp_neg_gamma1000": lambda x: np.exp(-x / 1000),
    "rbf": lambda x: np.exp(-x ** 2),
    "rbf_gamma_001": lambda x: np.exp(-x ** 2 / 0.001),
    "rbf_gamma_01": lambda x: np.exp(-x ** 2 / 0.01),
    "rbf_gamma_1": lambda x: np.exp(-x ** 2 / 0.1),
    "rbf_gamma10": lambda x: np.exp(-x ** 2 / 10),
    "rbf_gamma100": lambda x: np.exp(-x ** 2 / 100),
    "rbf_gamma1000": lambda x: np.exp(-x ** 2 / 1000),
    "quadratic": lambda x: x ** 2,
    "qudratic_1_1" : lambda x: 1 * (x + 1) ** 2,
    "quadratic_01_1": lambda x: 0.1 * (x + 1) ** 2,
    "quadratic_1_01": lambda x: 1 * (x + 0.1) ** 2,
    "quadratic_01_01": lambda x: 0.1 * (x + 0.1) ** 2,
    "multiquadric": lambda x: np.sqrt(x ** 2),
    "multiquadric_1": lambda x: np.sqrt(1 + x ** 2),
    "multiquadric_01": lambda x: np.sqrt(0.1 + x ** 2),
    "multiquadric_001": lambda x: np.sqrt(0.01 + x ** 2),
}
kernel_dict = {
    "linear": linear_kernel,
    "rbf": rbf_kernel,
    "rbf_gamma_001": lambda x1, x2: rbf_kernel(x1, x2, gamma=0.001),
    "rbf_gamma_01": lambda x1, x2: rbf_kernel(x1, x2, gamma=0.01),
    "rbf_gamma_1": lambda x1, x2: rbf_kernel(x1, x2, gamma=0.1),
    "rbf_gamma10": lambda x1, x2: rbf_kernel(x1, x2, gamma=10),
    "rbf_gamma100": lambda x1, x2: rbf_kernel(x1, x2, gamma=100),
    "rbf_gamma1000": lambda x1, x2: rbf_kernel(x1, x2, gamma=1000),
    "polynomial": polynomial_kernel,
    "polynomial_2": lambda x1, x2: polynomial_kernel(x1, x2, degree=2),
    "polynomial_3": lambda x1, x2: polynomial_kernel(x1, x2, degree=3),
    "polynomial_4": lambda x1, x2: polynomial_kernel(x1, x2, degree=4),
    "polynomial_5": lambda x1, x2: polynomial_kernel(x1, x2, degree=5),
}

def distance_helper(
    x1,
    x2,
    cx1=None,
    cx2=None,
    method="gzip",
    modified=False,
    symmetric=False,
):
    x1 = str(x1)
    x2 = str(x2)
    if modified is True and x1 == x2:
        return 0
    if modified is True and symmetric is True:
        if x1 >= x2:
            if method in compressors.keys():
                result = ncd(x1, x2, cx1, cx2, method)
            elif method in string_metrics.keys():
                result = calculate_string_distance(x1, x2, method)
            else:
                raise NotImplementedError(
                    f"Method {method} not supported. Supported methods are: {string_metrics.keys()} and {compressors.keys()}",
                )
        else:  # If x1 < x2, then swap the order
            if method in compressors.keys():
                result = ncd(x2, x1, cx2, cx1, method)
            elif method in string_metrics.keys():
                result = calculate_string_distance(x2, x1, method)
            else:
                raise NotImplementedError(
                    f"Method {method} not supported. Supported methods are: {string_metrics.keys()} and {compressors.keys()}",
                )
    elif (
        modified is False
    ):  # If not modified, then calculate the distance normally, without swapping or returning 0 when x1 == x2
        if method in compressors.keys():
            result = ncd(x1, x2, cx1, cx2, method)
        elif method in string_metrics.keys():
            result = calculate_string_distance(x1, x2, method)
        else:
            raise NotImplementedError(
                f"Method {method} not supported. Supported methods are: {string_metrics.keys()} and {compressors.keys()}",
            )
    elif modified is True and symmetric is False:
        if method in compressors.keys():
            result1 = ncd(x1, x2, cx1, cx2, method)
            result2 = ncd(x2, x1, cx2, cx1, method)
            result = (result1 + result2) / 2
        elif method in string_metrics.keys():
            result1 = calculate_string_distance(x1, x2, method)
            result2 = calculate_string_distance(x2, x1, method)
            result = (result1 + result2) / 2
        else:
            raise NotImplementedError(
                f"Method {method} not supported. Supported methods are: {string_metrics.keys()} and {compressors.keys()}",
            )
    else:
        print(f"Modified: {modified}, Symmetric: {symmetric}")
        print(f"type modified: {type(modified)}, type symmetric: {type(symmetric)}")
        raise ValueError(f"Expected {modified} and {symmetric} to be boolean")
    return result


def ncd(
    x1,
    x2,
    cx1=None,
    cx2=None,
    method: Literal["gzip", "lzma", "bz2", "zstd", "pkl", "brotli", None] = "gzip",
) -> float:
    """
    Calculate the normalized compression distance between two objects treated as strings.
    Args:
        x1 (str): The first object
        x2 (str): The second object
    Returns:
        float: The normalized compression distance between x1 and x2
    """

    compressor_len = (
        compressors[method] if method in compressors.keys() else compressors["gzip"]
    )
    x1 = str(x1) if not isinstance(x1, str) else x1
    x2 = str(x2) if not isinstance(x2, str) else x2
    Cx1 = compressor_len(x1) if cx1 is None else cx1
    Cx2 = compressor_len(x2) if cx2 is None else cx2
    x1x2 = "".join([x1, x2])
    Cx1x2 = compressor_len(x1x2)
    min_ = min(Cx1, Cx2)
    max_ = max(Cx1, Cx2)
    ncd = (Cx1x2 - min_) / max_
    return ncd


def calculate_string_distance(x1, x2, method):
    if method in string_metrics.keys():
        dist = string_metrics[method]
    else:
        raise NotImplementedError(
            f"Method {method} not supported. Supported methods are: {string_metrics.keys()}",
        )
    x1 = str(x1)
    x2 = str(x2)
    return dist(x1, x2)


class GzipClassifier(ClassifierMixin, BaseEstimator):
    """An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    k : int, default=3
        The number of neighbors to use.
    m: int, default=-1
        The number of best samples to use. If -1, all samples will be used.
    compressor: str, default="gzip"
        The name of the compressor to use. Choices are
    metric: str, default="ncd"
        The metric used to calculate the distance between samples. Choices are "gzip", "lzma", "bz2", "zstd", "pkl", "pickle", "levenshtein", "ratio", "seqratio", "hamming", "jaro", "jaro".
    transform: str, default=None
        The transformation to apply to the distance matrix. Choices are "abs", "log", "sqrt", "square", "exp", "exp_neg", "log_neg", "rbf".
    distance_matrix_train: str or np.ndarray, default=None
        The path to a numpy file or a numpy array representing the distance matrix. If a path is provided, the file will be loaded. If an array is provided, it will be used directly. Default is None.
    distance_matrix_test: str or np.ndarray, default=None
        The path to a numpy file or a numpy array representing the distance matrix. If a path is provided, the file will be loaded. If an array is provided, it will be used directly. Default is None.
    kernel: str, default=None
        The kernel to use to calculate kernel features from the distance matrix. Choices are "linear", "rbf", "polynomial".
    Attributes
    anchor : bool or None (default=None)
        If True, the first half of the training data will be used as the anchor. If False, the second half will be used as the anchor. If None, the anchor will not be used.
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    distance_matrix_ : ndarray, shape (n_samples, n_samples)
    """

    def __init__(
        self,
        distance_matrix_train=None,
        distance_matrix_test=None,
        metric="gzip",
        symmetric=False,
        modified=False,
        transform=None,
        kernel=None,
        anchor=None,
        n_jobs=-1,
        **kwargs,
    ):
        """
        Initialize the GzipClassifier object.

        Args:
            k (int): The value of k for k-nearest neighbors. Default is 3.
            m (int): The value of m for  m-best samples. Default is -1, which indicates using all training samples.
            metric (str): The metric used to calculate the distance between samples. Default is "ncd".
            distance_matrix (str or np.ndarray): The path to a numpy file or a numpy array representing the distance matrix.
                If a path is provided, the file will be loaded. If an array is provided, it will be used directly.
                Default is None.
            symmetric (bool): If True, the distance matrix will be treated as symmetric. Default is False.
            modified (bool): If True, inputs to ncd() will be sorted and ncd(x,x) will return 0. Default is False.
            transform (str): The transformation to apply to the distance matrix. These are kernel functions, but that is already a kwarg argument for SVCs, so something else had to be used. Default is None.
        Raises:
            ValueError: If distance_matrix is not a path to a numpy file or a numpy array.
            NotImplementedError: If the metric is not supported.
        """
        kwarg_string = str([f"{key}={value}" for key, value in kwargs.items()])
        logger.debug(
            f"Initializing GzipClassifier with distance_matrix_train={distance_matrix_train} distance_matrix_test={distance_matrix_test} metric={metric}, symmetric={symmetric}, {kwarg_string}",
        )
        if metric in compressors.keys():
            logger.debug(f"Using NCD metric with {metric} compressor.")
            self._distance = ncd
            self.metric = metric
        elif metric in string_metrics.keys():
            logger.debug(f"Using {metric} metric")
            self._distance = calculate_string_distance
            self.metric = metric
        else:
            raise NotImplementedError(
                f"Metric {metric} not supported. Supported metrics are: ncd, {string_metrics.keys()} and {compressors.keys()}",
            )
        assert modified in [
            True,
            False,
            None,
        ], f"Expected {self.modified} in [True, False, None]"
        transform_list = list(transform_dict.keys())
        transform_list.extend([None])
        if transform in [None, "None", "null", ""]:
            transform = None
        assert (
            transform in transform_list
        ), f"Expected {transform} in {transform_dict.keys()}"
        self.modified = False if modified is not True else True
        assert symmetric in [
            True,
            False,
            None,
        ], f"Expected {symmetric} in [True, False, None]"
        self.symmetric = symmetric
        self.transform = transform
        self.anchor = anchor
        if self.symmetric is True:
            self._calculate_training_distance_matrix = (
                self._calculate_lower_triangular_distance_matrix
            )
        elif symmetric in ["avg", "average"]:
            self._calculate_training_distance_matrix = (
                self._calculate_avg_with_transpose_distance_matrix
            )
        else:
            self._calculate_training_distance_matrix = (
                self._calculate_rectangular_distance_matrix
            )
        self.distance_matrix_train = distance_matrix_train
        self.distance_matrix_test = distance_matrix_test
        self.n_jobs = n_jobs
        kernel_list = list(kernel_dict.keys())
        kernel_list.extend([None, "precomputed", "None", "null", ""])
        assert kernel in kernel_list, f"Expected {kernel} in {kernel_list}. It is '{kernel}'"
        self.kernel = kernel
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _calculate_rectangular_distance_matrix(
        self,
        x1,
        x2,
        Cx1=None,
        Cx2=None,
        transform=None,
    ):
        """
        Calculate the distance matrix between two sets of objects, treating them as strings, assuming d(a,b) != d(b,a)
        Args:
            x1 (np.ndarray): The first set of objects
            x2 (np.ndarray): The second set of objects
        Returns:
            np.ndarray: The distance matrix of size (len(x1), len(x2))
        """
        n_jobs = self.n_jobs
        logger.info(f"Calculating rectangular distance matrix with {n_jobs} jobs")
        matrix_ = np.zeros((len(x1), len(x2)))
        Cx1 = Cx1 if Cx1 is not None else [None] * len(x1)
        Cx2 = Cx2 if Cx2 is not None else [None] * len(x2)
        list_ = []
        if n_jobs == -1:
            n_jobs = cpu_count()
        else:
            assert isinstance(n_jobs, int), f"Expected {n_jobs} to be an integer"
            assert n_jobs > 0, f"Expected {n_jobs} > 0 or -1"
            assert n_jobs <= cpu_count(), f"Expected {n_jobs} <= {cpu_count()}"
        for i in range(len(x1)):
            for j in range(len(x2)):
                list_.append((x1[i], x2[j], Cx1[i], Cx2[j]))
        list_ = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(distance_helper)(
                    *args,
                    modified=self.modified,
                    method=self.metric,
                    symmetric=self.symmetric,
                )
                for args in tqdm(
                    list_,
                    desc="Calculating rectangular distance matrix",
                    leave=False,
                    dynamic_ncols=True,
                    total=len(list_),
                )
            ),
        )
        matrix_ = list_.reshape(len(x1), len(x2))
        assert matrix_.shape == (
            len(x1),
            len(x2),
        ), f"Expected {matrix_.shape} == ({len(x1)}, {len(x2)})"
        if transform is not None:
            matrix_ = transform_dict[transform](matrix_)
        return matrix_

    def _calculate_lower_triangular_distance_matrix(
        self,
        x1,
        x2,
        Cx1=None,
        Cx2=None,
    ):
        """
        Calculate the distance matrix between two sets of objects, treating them as strings. Assuming the d(a,b) = d(b,a)
        Args:
            x1 (np.ndarray): The first set of objects
            x2 (np.ndarray): The second set of objects
        Returns:
            np.ndarray: The distance matrix of size (len(x1), len(x2))
        """
        n_jobs = self.n_jobs
        logger.info(f"Calculating lower triangular distance matrix with {n_jobs} jobs")
        # assert len(x1) == len(x2), f"Expected {len(x1)} == {len(x2)}"
        matrix_ = np.zeros((len(x1), len(x2)))
        list_ = []
        # Find the length of the longest list
        Cx1 = Cx1 if Cx1 is not None else [None] * len(x1)
        Cx2 = Cx2 if Cx2 is not None else [None] * len(x2)
        if n_jobs == -1:
            n_jobs = cpu_count()
        else:
            assert isinstance(n_jobs, int), f"Expected {n_jobs} to be an integer"
            assert n_jobs > 0, f"Expected {n_jobs} > 0 or -1"
            assert n_jobs <= cpu_count(), f"Expected {n_jobs} <= {cpu_count()}"
        # Create a list of tuples to pass to the parallel function
        for i in range(len(x1)):
            for j in range(0, i + 1):
                list_.append((x1[i], x2[j], Cx1[i], Cx2[j]))
        list_ = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(distance_helper)(
                    *args,
                    modified=self.modified,
                    method=self.metric,
                    symmetric=self.symmetric,
                )
                for args in tqdm(
                    list_,
                    desc="Calculating symmetric distance matrix",
                    leave=False,
                    dynamic_ncols=True,
                    total=len(list_),
                )
            ),
        )
        indices = np.tril_indices(len(x1))
        matrix_[indices] = list_
        old_diag = np.diag(np.diag(matrix_))
        # Add matrix to its transpose and subtract the diagonal to avoid double counting
        matrix_ = matrix_ + matrix_.T - old_diag
        new_diag = np.diag(np.diag(matrix_))
        # Check that old_diag is close to new_diag
        assert np.allclose(
            old_diag,
            new_diag,
        ), f"Expected {old_diag} == {new_diag}. Old Diag: {old_diag}"
        # Check the shape of the matrix
        assert matrix_.shape == (
            len(x1),
            len(x2),
        ), f"Expected {matrix_.shape} == ({len(x1)}, {len(x2)}). "
        return matrix_

    def _calculate_upper_triangular_distance_matrix(
        self,
        x1,
        x2,
        Cx1=None,
        Cx2=None,
    ):
        n_jobs = self.n_jobs
        logger.info(f"Calculating upper triangular distance matrix with {n_jobs} jobs")
        matrix_ = np.zeros((len(x1), len(x2)))
        Cx1 = Cx1 if Cx1 is not None else [None] * len(x1)
        Cx2 = Cx2 if Cx2 is not None else [None] * len(x2)
        list_ = []
        for i in range(len(x1)):
            for j in range(i, len(x2)):
                list_.append((x1[i], x2[j], Cx1[i], Cx2[j]))
        list_ = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(distance_helper)(
                    *args,
                    modified=self.modified,
                    method=self.metric,
                    symmetric=self.symmetric,
                )
                for args in list_
            ),
        )
        indices = np.triu_indices(len(x1))
        matrix_[indices] = list_
        old_diag = np.diag(np.diag(matrix_))
        # Add matrix to its transpose and subtract the diagonal to avoid double counting
        matrix_ = matrix_ + matrix_.T - old_diag
        new_diag = np.diag(np.diag(matrix_))
        # Check that old_diag is close to new_diag
        assert np.allclose(
            old_diag,
            new_diag,
        ), f"Expected {old_diag} == {new_diag}. Old Diag: {old_diag}"
        # Check the shape of the matrix
        assert matrix_.shape == (
            len(x1),
            len(x2),
        ), f"Expected {matrix_.shape} == ({len(x1)}, {len(x2)})"

        return matrix_

    def _load_distance_matrix(self, path):
        if Path(path).exists():
            path = Path(path).resolve().as_posix()
            with open(path, "rb") as f:
                matrix = np.load(f)["X"]
        else:
            raise FileNotFoundError(f"Distance matrix file {path} not found")
        return matrix

    def _save_distance_matrix(self, path, matrix):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez_compressed(f, X=matrix)

    def _prepare_training_distance_matrix(self):
        """
        Prepare the distance matrix for classification.
        If self.distance_matrix is a path to a numpy file, it will be loaded.
        If it is a numpy array, it will be used directly.
        If it is None, the distance matrix will be calculated using self.X_ and self.X_.
        """
        n_jobs = self.n_jobs
        if self.anchor is True:
            self._prepare_anchor()
            X2 = self.X_
            X1 = self.X2_
        else:
            X1 = self.X_
            X2 = self.X_
        logger.info(f"Preparing training matrix with {n_jobs} jobs")
        if self.metric in compressors.keys():
            compressor = compressors[self.metric]
            Cx_ = Parallel(n_jobs=n_jobs)(delayed(compressor)(x) for x in X1)
            Cx1 = np.array(Cx_) if not isinstance(Cx_, np.ndarray) else Cx_
            self.Cx_ = Cx1
            Cx_ = Parallel(n_jobs=n_jobs)(delayed(compressor)(x) for x in X2)
            Cx2 = np.array(Cx_) if not isinstance(Cx_, np.ndarray) else Cx_
        else:
            self.Cx_ = None
            Cx1 = self.Cx_
            Cx2 = self.Cx_
            self.X_ = self.X_.astype(str)
        if (
            self.distance_matrix_train is not None
            and Path(self.distance_matrix_train).exists()
        ):
            distance_matrix = self._load_distance_matrix(self.distance_matrix_train)
        else:
            distance_matrix = self._calculate_training_distance_matrix(
                X1,
                X2,
                Cx1,
                Cx2,
            )
        assert (
            distance_matrix.shape[0] == distance_matrix.shape[1]
        ), f"Distance matrix must be square, got {distance_matrix.shape}"
        assert (
            len(X1) == distance_matrix.shape[0]
        ), f"Expected {len(X1)} == {distance_matrix.shape[0]}"
        assert (
            len(self.y_) == distance_matrix.shape[0]
        ), f"Expected len(y) == {distance_matrix.shape[0]}"
        if isinstance(self.distance_matrix_train, (str, Path)):
            # Save the distance matrix
            self._save_distance_matrix(self.distance_matrix_train, distance_matrix)
        return distance_matrix

    def _prepare_anchor(self):
        # Split the data into two halves
        N = len(self.X_) // 2
        X1, X2, y1, y2 = train_test_split(
            self.X_,
            self.y_,
            train_size=N,
            test_size=N,
            random_state=42,
            stratify=self.y_,
        )
        assert X1.shape == X2.shape, f"Expected {X1.shape} == {X2.shape}"
        assert y1.shape == y2.shape, f"Expected {y1.shape} == {y2.shape}"
        self.X_ = X1
        self.X2_ = X2
        self.y_ = y1
        self.y2_ = y2

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Fit the model using X as training data and y as target values.

        Args:
            X (np.ndarray): The input data
            y (np.ndarray): The target labels
            n_jobs (int): The number of jobs to run in parallel. Default is -1.

        Returns:
            GzipClassifier: The fitted model
        """
        # The code snippet is performing an assertion check in Python. It is verifying whether the
        # length of the list `X` is equal to the length of the list `y`. If the lengths are not equal,
        # it will raise an AssertionError with a message indicating the expected and actual lengths.
        # assert len(X) == len(y), f"Expected {len(X)} == {len(y)}"
        
        logger.debug(f"Fitting with X of shape {X.shape} and y of shape {y.shape}")
        self.y_ = self._prepare_y(y)
        self.X_ =  self._prepare_X(X)
        self._train_matrix = self._prepare_training_distance_matrix()
        self._train_matrix = self._transform_training_matrix(self._train_matrix)
        self.clf_ = self.clf_.fit(self._train_matrix, self.y_)
        return self

    def _transform_training_matrix(self, training_matrix):
        if self.kernel not in [None, "precomputed", "None", "null", ""]:
            kernel = kernel_dict[self.kernel]
            training_matrix = kernel(training_matrix, training_matrix)
        if self.transform is not None:
            training_matrix = transform_dict[self.transform](training_matrix)
        return training_matrix

    def _prepare_X(self, X):
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        X = np.array([str(x) for x in X])
        self.n_features_ = X.shape[1] if len(X.shape) > 1 else 1
        return X

    def _prepare_y(self, y):
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        if len(np.squeeze(y).shape) == 1:
            encoder = LabelBinarizer()
            y = encoder.fit_transform(y)
            self.n_classes_ = len(unique_labels(y))
            flat_y = np.squeeze(y).astype(int)
        else:
            y = y
            self.n_classes_ = y.shape[1]
            flat_y = np.argmax(y, axis=1)
        counts = np.bincount(flat_y)
        self.counts_ = counts
        logger.debug(f"Num Classes: {self.n_classes_}, counts: {counts}")
        self.classes_ = range(len(unique_labels(y)))
        return y

    def predict(self, X: np.ndarray):
        """Predict the class labels for the provided data.

        Args:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted class labels
        """
        check_is_fitted(self)
        logger.debug(f"Predicting with X of shape {X.shape}")
        X = self._prepare_X(X)
        distance_matrix = self._prepare_test_matrix(X)
        distance_matrix = self._transform_test_matrix(distance_matrix)
        y_pred = self.clf_.predict(distance_matrix)
        y_pred = self._format_predictions(y_pred)
        return y_pred

    def _transform_test_matrix(self, distance_matrix):
        if self.kernel not in [None, "precomputed", "None", "null", ""]:
            kernel = kernel_dict[self.kernel]
            distance_matrix = kernel(distance_matrix, self._train_matrix)
        if self.transform is not None:
            distance_matrix = transform_dict[self.transform](distance_matrix)
        return distance_matrix

    def _format_predictions(self, y_pred):
        if len(np.squeeze(y_pred).shape) == 1:
            encoder = LabelBinarizer()
            y_pred = encoder.fit(self.y_).transform(y_pred)
        else:
            encoder = LabelEncoder()
            y_pred = encoder.fit(self.y_).transform(y_pred)
        return y_pred

    def _prepare_test_matrix(self, X):
        if (
            self.distance_matrix_test is not None
            and Path(self.distance_matrix_test).exists()
        ):
            distance_matrix = self._load_distance_matrix(self.distance_matrix_test)
        else:
            if self.metric in compressors.keys():
                compressor = compressors[self.metric]
                Cx2 = Parallel(n_jobs=self.n_jobs)(
                    delayed(compressor)(x)
                    for x in tqdm(
                        X,
                        desc="Compressing samples",
                        leave=False,
                        dynamic_ncols=True,
                    )
                )
                # assert len(Cx2) == len(X), f"Expected {len(Cx2)} == {len(X)}"
                # assert len(self.X_) == len(
                #     self.Cx_,
                # ), f"Expected {len(self.X_)} == {len(self.Cx_)}"
                distance_matrix = self._calculate_rectangular_distance_matrix(
                    x1=X,
                    Cx1=Cx2,
                    x2=self.X_,
                    Cx2=self.Cx_,
                    transform=self.transform,
                )
            else:
                distance_matrix = self._calculate_rectangular_distance_matrix(
                    x2=self.X_,
                    x1=X,
                    transform=self.transform,
                )
        if self.distance_matrix_test is not None:
            # Save the distance matrix
            self._save_distance_matrix(self.distance_matrix_test, distance_matrix)
        return distance_matrix

    def score(self, X: np.ndarray, y: np.ndarray):
        """Score the model using the provided data.

        Args:
            X (np.ndarray): The input data
            y (np.ndarray): The target labels

        Returns:
            float: The accuracy of the model
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class GzipKNN(GzipClassifier):
    def __init__(
        self,
        k: int = 2,
        distance_matrix_train=None,
        distance_matrix_test=None,
        metric="gzip",
        weights=None,
        symmetric=False,
        modified=False,
        transform=None,
        anchor=None,
        n_jobs=-1,
        kernel = None,
        **kwargs,
    ):
        super().__init__(
            distance_matrix_train=distance_matrix_train,
            distance_matrix_test=distance_matrix_test,
            metric=metric,
            symmetric=symmetric,
            modified=modified,
            transform=transform,
            anchor=anchor,
            n_jobs=n_jobs,
            weights=weights,
            kernel=kernel,
            **kwargs,
        )
        self.clf_ = KNeighborsClassifier(
            n_neighbors=kwargs.pop("n_neighbors", k),
            metric="precomputed",
            **kwargs,
        )
        self.k = k
        for k, v in kwargs.items():
            setattr(self, k, v)
    
        
    

class GzipLogisticRegressor(GzipClassifier):
    def __init__(
        self,
        distance_matrix_train=None,
        distance_matrix_test=None,
        metric="gzip",
        symmetric=False,
        modified=False,
        transform=None,
        n_jobs=-1,
        anchor=None,
        penalty=None,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        kernel=None,
        **kwargs,
    ):
        clf = LogisticRegression(**kwargs)
        super().__init__(
            clf_=clf,
            distance_matrix_train=distance_matrix_train,
            distance_matrix_test=distance_matrix_test,
            metric=metric,
            symmetric=symmetric,
            modified=modified,
            anchor=anchor,
            n_jobs=n_jobs,
            transform=transform,
            penalty=penalty,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            kernel=kernel,
            **kwargs,
        )
        for k, v in kwargs.items():
            setattr(self, k, v)


class GzipSVC(GzipClassifier):
    def __init__(
        self,
        m=0,
        distance_matrix_train=None,
        distance_matrix_test=None,
        metric="gzip",
        symmetric=False,
        modified=False,
        anchor=None,
        transform=None,
        C=1.0,
        tol=1e-3,
        n_jobs=-1,
        **kwargs,
    ):
        if "kernel" not in kwargs.keys():
            kwargs["kernel"] = "precomputed"
        elif kwargs["kernel"] is None:
            kwargs["kernel"] = "precomputed"
        clf = SVC(**kwargs, C=C, tol=tol)
        super().__init__(
            clf_=clf,
            m=m,
            distance_matrix_train=distance_matrix_train,
            distance_matrix_test=distance_matrix_test,
            metric=metric,
            symmetric=symmetric,
            modified=modified,
            anchor=anchor,
            transform=transform,
            n_jobs=n_jobs,
            tol=tol,
            C=C,
            **kwargs,
        )
        for k, v in kwargs.items():
            setattr(self, k, v)


class GridSearchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        n_jobs=-1,
        cv=None,
        verbose=0,
        refit=True,
        return_train_score=False,
        **kwargs,
    ):
        estimator = eval(estimator)
        assert isinstance(
            estimator,
            BaseEstimator,
        ), f"Expected {estimator} to be a BaseEstimator"
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose
        self.refit = refit
        self.return_train_score = return_train_score
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, X, y):
        self.clf_ = GridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
            verbose=self.verbose,
            refit=self.refit,
            return_train_score=self.return_train_score,
        )
        self.clf_ = self.clf_.fit(X, y)
        return self.clf_

    def predict(self, X):
        return self.clf_.predict(X)

    def score(self, X, y):
        return self.clf_.score(X, y)

    def get_params(self, deep=True):
        return self.clf_.get_params(deep=deep)

    def set_params(self, **params):
        return self.clf_.set_params(**params)


supported_models = {
    "knn": GzipKNN,
    "logistic": GzipLogisticRegressor,
    "svc": GzipSVC,
}


model_scorers = {
    "knn": "accuracy",
    "logistic": "accuracy",
    "svc": "accuracy",
}

scorers = {
    "accuracy": accuracy_score,
}


def test_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_type,
    optimizer=None,
    **kwargs,
) -> dict:
    """
    Args:
        X_train (np.ndarray): The input data
        X_test (np.ndarray): The test data
        y_train (np.ndarray): The target labels
        y_test (np.ndarray): The test labels
        model_type (str): The type of model to use. Choices are "knn", "logistic", "svc".
        optimizer (str): The metric to optimize. Choices are "accuracy", "f1", "precision", "recall".
        **kwargs: Additional keyword arguments to pass to the GzipClassifier
    Returns:
        dict: A dictionary containing the accuracy, train_time, and pred_time
    """
    model = supported_models[model_type](**kwargs)
    alias = model_scorers[model_type]
    scorer = scorers[alias]
    start = time.time()

    model.fit(X_train, y_train)
    check_is_fitted(model)
    end = time.time()
    train_time = end - start
    start = time.time()
    predictions = model.predict(X_test)
    end = time.time()
    pred_time = end - start
    score = round(scorer(y_test, predictions), 3)
    print(f"Training time: {train_time}")
    print(f"Prediction time: {pred_time}")
    print(f"{alias.capitalize()}  is: {score}")
    score_dict = {
        f"{alias.lower()}": score,
        "train_time": train_time,
        "pred_time": pred_time,
    }
    if optimizer is not None:
        score = score_dict[optimizer]
        return score
    else:
        return score_dict


def load_data(dataset, precompressed):
    if dataset == "20newsgroups":
        X, y = fetch_20newsgroups(
            subset="train",
            categories=["alt.atheism", "talk.religion.misc"],
            shuffle=True,
            random_state=42,
            return_X_y=True,
        )
        y = (
            LabelEncoder().fit(y).transform(y)
        )  # Turns the labels "alt.atheism" and "talk.religion.misc" into 0 and 1
    elif dataset == "kdd_nsl":
        df = pd.read_csv("raw_data/kdd_nsl_undersampled_5000.csv")
        y = df["label"]
        X = df.drop("label", axis=1)
    elif dataset == "make_classification":
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2,
            random_state=42,
        )
        y = LabelEncoder().fit(y).transform(y)
    elif dataset == "truthseeker":
        df = pd.read_csv("raw_data/truthseeker_undersampled_8000.csv")
        y = df["BotScoreBinary"]
        X = df.drop("BotScoreBinary", axis=1)
    elif dataset == "sms-spam":
        df = pd.read_csv("raw_data/sms-spam_undersampled_1450.csv")
        y = df["label"]
        X = df.drop("label", axis=1)
    elif dataset == "ddos":
        df = pd.read_csv("raw_data/ddos.csv")
        y = df["Label"]
        X = df.drop("Label", axis=1)
    else:
        raise ValueError(
            f"Dataset {dataset} not found. Options are: 20newsgroups, kdd_nsl, make_classification, truthseeker, sms-spam, ddos.",
        )
    if precompressed is True:
        X = pd.DataFrame(X).applymap(lambda x: len(gzip.compress(str(x).encode())))
    else:
        X = pd.DataFrame(X).applymap(str)
    X = np.array(X)
    return X, y


def prepare_data(
    dataset="truthseeker",
    precompressed=False,
    train_size=100,
    test_size=100,
    random_state=42,
):
    X, y = load_data(dataset, precompressed=precompressed)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    # Make into data
    return X_train, X_test, y_train, y_test


def main(args: argparse.Namespace):
    """
    This is the main function that runs the GzipClassifier with the provided arguments.
    It will fetch the dataset, split it into training and testing sets.
    Then, it will train the model using the fit method and test it using the predict method.
    Args:
        args (argparse.Namespace): The command line arguments
    Usage:
        python gzip_classifier.py --metric gzip  --m 10  --dataset kdd_nsl k=3
    """

    X, y = load_data(dataset=args.dataset, precompressed=args.precompressed)
    params = vars(args)
    dataset = params.pop("dataset")
    precompressed = params.pop("precompressed")
    train_size = params.pop("train_size")
    test_size = params.pop("test_size")
    random_state = params.pop("random_state")
    X_train, X_test, y_train, y_test = prepare_data(
        dataset=dataset,
        precompressed=precompressed,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )
    kwarg_args = params.pop("kwargs")
    # conver list of key-value pairs to dictionary
    kwarg_args = dict([arg.split("=") for arg in kwarg_args])
    for k, v in kwarg_args.items():
        # Typecast the values to the correct type
        try:
            kwarg_args[k] = eval(v)
        except:  # noqa E722
            kwarg_args[k] = v
    params.update(**kwarg_args)
    params.pop("cross_validate")
    params.pop("grid_search")
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    test_model(X_train, X_test, y_train, y_test, **params)


def cross_validate_main(args: argparse.Namespace):
    """
    This is the main function that runs the GzipClassifier with the provided arguments.
    It will fetch the dataset, split it into training and testing sets.
    Then, it will train the model using the fit method and test it using the predict method.
    Args:
        args (argparse.Namespace): The command line arguments
    Usage:
        python gzip_classifier.py --metric gzip  --m 10  --dataset kdd_nsl k=3
    """
    X, y = load_data(dataset=args.dataset, precompressed=args.precompressed)
    params = vars(args)
    dataset = params.pop("dataset")
    precompressed = params.pop("precompressed")
    train_size = params.pop("train_size")
    test_size = params.pop("test_size")
    random_state = params.pop("random_state")
    X_train, X_test, y_train, y_test = prepare_data(
        dataset=dataset,
        precompressed=precompressed,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )
    # StratifiedKFold

    kwarg_args = params.pop("kwargs")
    # convert list of key-value pairs to dictionary
    kwarg_args = dict([arg.split("=") for arg in kwarg_args])
    for k, v in kwarg_args.items():
        # Typecast the values to the correct type
        try:
            kwarg_args[k] = eval(v)
        except:  # noqa E722
            kwarg_args[k] = v
    params.update(**kwarg_args)
    params.pop("cross_validate")
    params.pop("grid_search")
    model_type = params.pop("model_type")
    optimizer = params.pop("optimizer")
    skf = StratifiedKFold(
        n_splits=params.pop("n_splits", 5),
        random_state=random_state,
        shuffle=True,
    )
    model = supported_models[model_type](**params)
    cv_scores = cross_validate(
        X=X_train,
        y=y_train,
        cv=skf,
        estimator=model,
        scoring=optimizer,
        n_jobs=1,
    )
    print(f"mean of cross-validation scores: {cv_scores['test_score'].mean()}")
    print(f"std of cross-validation scores: {cv_scores['test_score'].std()}")
    # Validate the model using the withheld test data
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test score: {score}")


def grid_search_main(args: argparse.Namespace):
    """
    This is the main function that runs the GzipClassifier with the provided arguments.
    It will fetch the dataset, split it into training and testing sets.
    Then, it will train the model using the fit method and test it using the predict method.
    Args:
        args (argparse.Namespace): The command line arguments
    Usage:
        python gzip_classifier.py --metric gzip  --m 10  --dataset kdd_nsl k=3
    """
    X, y = load_data(dataset=args.dataset, precompressed=args.precompressed)
    params = vars(args)
    dataset = params.pop("dataset")
    precompressed = params.pop("precompressed")
    train_size = params.pop("train_size")
    test_size = params.pop("test_size")
    random_state = params.pop("random_state")
    X_train, X_test, y_train, y_test = prepare_data(
        dataset=dataset,
        precompressed=precompressed,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )
    # StratifiedKFold
    kwarg_args = params.pop("kwargs")
    # conver list of key-value pairs to dictionary
    kwarg_args = dict([arg.split("=") for arg in kwarg_args])
    n_splits = eval(kwarg_args.pop("n_splits", 5))
    for k, v in kwarg_args.items():
        # Turn all values into lists
        try:
            v = eval(v)
        except:  # noqa E722
            v = str(v)
            vs = v.split(",")
            for i in range(len(vs)):
                try:
                    vs[i] = eval(vs[i])
                except:  # noqa E722
                    vs[i] = str(vs[i])
            v = vs
        if isinstance(v, tuple):
            v = list(v)
        elif not isinstance(v, list):
            v = [v]
        kwarg_args[k] = v

    params.update(**kwarg_args)
    params.pop("cross_validate")
    params.pop("grid_search")
    model_type = params.pop("model_type")
    optimizer = params.pop("optimizer")
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    n_jobs = params.pop("n_jobs", cpu_count())
    model = supported_models[model_type](n_jobs=1)
    # Assume that kwarg_args contains the hyperparameters to search
    grid = GridSearchCV(
        estimator=model,
        param_grid=kwarg_args,
        cv=skf,
        scoring=optimizer,
        n_jobs=n_jobs,
        verbose=3,
    )
    # Ravel the y_train
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    grid.fit(X_train, y_train)
    print(f"Best score: {grid.best_score_}")
    print(f"Best params: {grid.best_params_}")
    # Validate the model using the withheld test data
    score = grid.score(X_test, y_test)
    print(f"Test score: {score}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    default="knn",
    help="The type of model to use. Choices are knn, logistic, svc",
)
parser.add_argument(
    "--symmetric",
    action="store_true",
    help="If True, the distance matrix will be treated as symmetric. Default is False.",
)
parser.add_argument(
    "--metric",
    type=str,
    default="gzip",
    choices=all_metrics,
    help=f"The metric used to calculate the distance between samples. Choices are {list(all_metrics.keys())}",
)
parser.add_argument(
    "--distance_matrix_train",
    type=str,
    default=None,
    help="The path to a numpy array representing the distance matrix. If a path is provided, the file will be loaded. Default is None.",
)
parser.add_argument(
    "--distance_matrix_test",
    type=str,
    default=None,
    help="The path to a numpy array representing the distance matrix. If a path is provided, the file will be loaded. Default is None.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="kdd_nsl",
    help="The dataset to use. Choices are 20newsgroups, kdd_nsl, make_classification, truthseeker, sms-spam, ddos.",
)
parser.add_argument(
    "--train_size",
    type=int,
    default=100,
    help="The number of samples to use for training. Default is 100.",
)
parser.add_argument(
    "--test_size",
    type=int,
    default=100,
    help="The number of samples to use for testing. Default is 100.",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="accuracy",
    help="The metric to use for optimization. Default is accuracy.",
)
parser.add_argument(
    "--precompressed",
    action="store_true",
    help="If True, the data will be precompressed using gzip.",
)
parser.add_argument(
    "--random_state",
    type=int,
    default=42,
    help="The random state to use. Default is 42.",
)
parser.add_argument(
    "kwargs",
    nargs=argparse.REMAINDER,
    help="Additional keyword arguments to pass to the GzipClassifier",
)
parser.add_argument(
    "--cross_validate",
    action="store_true",
    help="If True, the model will be cross-validated using StratifiedKFold.",
)
parser.add_argument(
    "--grid_search",
    action="store_true",
    help="If True, the model will be cross-validated using GridSearchCV.",
)
if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.cross_validate is True:
        # pop cross_validate from the arguments
        args.cross_validate = None
        assert args.grid_search is False, f"Expected {args.grid_search} is None"
        cross_validate_main(args)
    elif args.grid_search is True:
        # pop grid_search from the arguments
        args.grid_search = None
        assert args.cross_validate is False, f"Expected {args.cross_validate} is None"
        grid_search_main(args)
    else:
        main(args)
