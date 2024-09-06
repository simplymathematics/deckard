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
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn_extra.cluster import KMedoids
from imblearn.under_sampling import (
    CondensedNearestNeighbour,
    NearMiss,
    InstanceHardnessThreshold,
)
from Levenshtein import distance, ratio, hamming, jaro, jaro_winkler, seqratio
import pandas as pd
from multiprocessing import cpu_count

from joblib import Parallel, delayed
from typing import Literal

from batchMixin import BatchedMixin

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


compressors = {
    "gzip": _gzip_len,
    "lzma": _lzma_len,
    "bz2": _bz2_len,
    "zstd": _zstd_len,
    "pkl": _pickle_len,
}


def ncd(
    x1,
    x2,
    cx1=None,
    cx2=None,
    method: Literal["gzip", "lzma", "bz2", "zstd", "pkl", None] = "gzip",
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
    x1 = str(x1)
    x2 = str(x2)
    Cx1 = compressor_len(x1) if cx1 is None else cx1
    Cx2 = compressor_len(x2) if cx2 is None else cx2
    x1x2 = " ".join([x1, x2])
    Cx1x2 = compressor_len(x1x2)
    min_ = min(Cx1, Cx2)
    max_ = max(Cx1, Cx2)
    ncd = (Cx1x2 - min_) / max_
    return ncd


string_metrics = {
    "levenshtein": distance,
    "ratio": ratio,
    "hamming": hamming,
    "jaro": jaro,
    "jaro_winkler": jaro_winkler,
    "seqratio": seqratio,
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
]


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
    method: str, default="random"
        The method used to select the best training samples. Choices are "sum", "mean", "medoid", "random", "knn", "svc".
    metric: str, default="ncd"
        The metric used to calculate the distance between samples. Choices are "gzip", "lzma", "bz2", "zstd", "pkl", "pickle", "levenshtein", "ratio", "seqratio", "hamming", "jaro", "jaro".
    distance_matrix: str or np.ndarray, default=None
        The path to a numpy file or a numpy array representing the distance matrix. If a path is provided, the file will be loaded. If an array is provided, it will be used directly. Default is None.
    Attributes
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
        m=0,
        condensing_method="random",
        distance_matrix=None,
        metric="gzip",
        symmetric=False,
        similarity=False,
        double_centering=False,
        min_max_scale=False,
        modified=False,
        **kwargs,
    ):
        """
        Initialize the GzipClassifier object.

        Args:
            k (int): The value of k for k-nearest neighbors. Default is 3.
            m (int): The value of m for  m-best samples. Default is -1, which indicates using all training samples.
            condensing_method (str): The method used for classification. Default is "random".
            metric (str): The metric used to calculate the distance between samples. Default is "ncd".
            distance_matrix (str or np.ndarray): The path to a numpy file or a numpy array representing the distance matrix.
                If a path is provided, the file will be loaded. If an array is provided, it will be used directly.
                Default is None.
            symmetric (bool): If True, the distance matrix will be treated as symmetric. Default is False.

        Raises:
            ValueError: If distance_matrix is not a path to a numpy file or a numpy array.
            NotImplementedError: If the metric is not supported.
        """
        kwarg_string = str([f"{key}={value}" for key, value in kwargs.items()])
        logger.debug(
            f"Initializing GzipClassifier with  m={m},  method={condensing_method}, distance_matrix={distance_matrix}, metric={metric}, symmetric={symmetric}, {kwarg_string}",
        )
        self.m = m
        self.similarity = similarity
        self.double_centering = double_centering
        self.min_max_scale = min_max_scale
        if self.m > 0:
            assert (
                condensing_method in all_condensers or condensing_method is None
            ), f"Expected {condensing_method} in {all_condensers}"
        self.condensing_method = condensing_method
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
        self.modified = False if modified is not True else True
        self.symmetric = False if symmetric is not True else True
        if self.symmetric is True:
            self._calculate_distance_matrix = (
                self._calculate_lower_triangular_distance_matrix
            )
        else:
            self._calculate_distance_matrix = (
                self._calculate_rectangular_distance_matrix
            )
        self.distance_matrix = distance_matrix
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _calculate_rectangular_distance_matrix(
        self,
        x1,
        x2,
        Cx1=None,
        Cx2=None,
        n_jobs=-1,
    ):
        """
        Calculate the distance matrix between two sets of objects, treating them as strings, assuming d(a,b) != d(b,a)
        Args:
            x1 (np.ndarray): The first set of objects
            x2 (np.ndarray): The second set of objects
        Returns:
            np.ndarray: The distance matrix of size (len(x1), len(x2))
        """
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
                delayed(self._distance_helper)(*args)
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
        return matrix_

    def _distance_helper(self, x1, x2, cx1=None, cx2=None):
        method = self.metric
        if self.modified is True and str(x1) == str(x2):
            return 0
        elif self.modified is True and str(x1) != str(x2):
            if str(x1) > str(x2):
                s1 = x1
                cs1 = cx1
                s2 = x2
                cs2 = cx2
            else:
                s1 = x2
                cs1 = cx2
                s2 = x1
                cs2 = cx1
        else:
            assert self.modified is False, f"Expected {self.modified} to be False"
        if method in compressors.keys():
            result = ncd(x1, x2, cs1, cs2, method)
        elif method in string_metrics.keys():
            result = calculate_string_distance(s1, s2, method)
        else:
            raise NotImplementedError(
                f"Method {method} not supported. Supported methods are: {string_metrics.keys()} and {compressors.keys()}",
            )
        return result

    def _calculate_lower_triangular_distance_matrix(
        self,
        x1,
        x2,
        Cx1=None,
        Cx2=None,
        n_jobs=-1,
    ):
        """
        Calculate the distance matrix between two sets of objects, treating them as strings. Assuming the d(a,b) = d(b,a)
        Args:
            x1 (np.ndarray): The first set of objects
            x2 (np.ndarray): The second set of objects
        Returns:
            np.ndarray: The distance matrix of size (len(x1), len(x2))
        """
        assert len(x1) == len(x2), f"Expected {len(x1)} == {len(x2)}"
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
                delayed(self._distance_helper)(*args)
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
        n_jobs=-1,
    ):
        matrix_ = np.zeros((len(x1), len(x2)))
        Cx1 = Cx1 if Cx1 is not None else [None] * len(x1)
        Cx2 = Cx2 if Cx2 is not None else [None] * len(x2)
        list_ = []
        for i in range(len(x1)):
            for j in range(i, len(x2)):
                list_.append((x1[i], x2[j], Cx1[i], Cx2[j]))
        list_ = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(self._distance_helper)(*args) for args in list_
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
            return np.load(path, allow_pickle=True)["X"]
        else:
            raise FileNotFoundError(f"Distance matrix file {path} not found")

    def _save_distance_matrix(self, path, matrix):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=matrix)

    def _prepare_training_matrix(self, n_jobs=-1, update=False):
        """
        Prepare the distance matrix for classification.
        If self.distance_matrix is a path to a numpy file, it will be loaded.
        If it is a numpy array, it will be used directly.
        If it is None, the distance matrix will be calculated using self.X_ and self.X_.
        """

        if (
            isinstance(self.distance_matrix, str)
            and Path(self.distance_matrix).exists()
        ):
            distance_matrix = self._load_distance_matrix(self.distance_matrix)
        elif (
            isinstance(self.distance_matrix, str)
            and not Path(self.distance_matrix).exists()
        ):
            distance_matrix = self._calculate_distance_matrix(
                self.X_,
                self.X_,
                Cx1=self.Cx_,
                Cx2=self.Cx_,
                n_jobs=n_jobs,
            )
            self._save_distance_matrix(self.distance_matrix, distance_matrix)
        elif isinstance(self.distance_matrix, np.ndarray) and update is False:
            distance_matrix = self.distance_matrix
        elif isinstance(self.distance_matrix, np.ndarray) and update is True:
            distance_matrix = self._calculate_distance_matrix(
                self.X_,
                self.X_,
                Cx1=self.Cx_,
                Cx2=self.Cx_,
                n_jobs=n_jobs,
            )
        elif isinstance(self.distance_matrix, type(None)):
            distance_matrix = self._calculate_distance_matrix(
                self.X_,
                self.X_,
                Cx1=self.Cx_,
                Cx2=self.Cx_,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(
                f"distance_matrix must be a path to a numpy file or a numpy array, got {type(self.distance_matrix)}",
            )
        assert (
            distance_matrix.shape[0] == distance_matrix.shape[1]
        ), f"Distance matrix must be square, got {distance_matrix.shape}"
        assert (
            len(self.X_) == distance_matrix.shape[0]
        ), f"Expected {len(self.X_)} == {distance_matrix.shape[0]}"
        assert (
            len(self.y_) == distance_matrix.shape[0]
        ), f"Expected len(y) == {distance_matrix.shape[0]}"
        # Convert from distance to similarity
        if self.similarity is True:
            max_ = np.max(distance_matrix)
            distance_matrix = max_ - distance_matrix  # Similarity = 1 - distance
        elif self.similarity is False:
            pass
        else:  # pragma: no cover
            raise NotImplementedError(
                f"Similarity {self.similarity} not supported. Supported similarities are: True, False",
            )
        # Min-max scale
        if self.min_max_scale is True:
            min_ = np.min(distance_matrix)
            max_ = np.max(distance_matrix)
            distance_matrix = (distance_matrix - min_) / (max_ - min_)

        return distance_matrix

    def _find_best_samples(self, method="medoid", n_jobs=-1, update=False):
        """
        Args:
            method (str): The method used to select the best training samples. Default is "medoid". Choices are "sum", "mean", "medoid", "random", "knn", "svc".
        Returns:
            list: The indices of the best training samples.
        """
        self.distance_matrix = self._prepare_training_matrix(n_jobs=n_jobs)
        assert isinstance(
            self.distance_matrix,
            np.ndarray,
        ), f"Expected {type(self.distance_matrix)} to be np.ndarray"
        distance_matrix = self.distance_matrix
        indices = []
        if isinstance(self.m, float):
            m = int(self.m * len(self.X_) / self.n_classes_)
            if m == 0:
                m = 1
        else:
            m = self.m
        y = self.y_
        n_classes = len(unique_labels(y))
        if method in ["sum", "medoid", "svc", "random"]:
            if method == "sum":
                for label in np.unique(y):
                    label_idx = np.where(y == label)[0]
                    label_distance_matrix = distance_matrix[label_idx, :]
                    summed_matrix = np.sum(label_distance_matrix, axis=0)
                    sorted_idx = np.argsort(summed_matrix)
                    indices.extend(sorted_idx[:m])
            elif method == "medoid":
                for label in np.unique(y):
                    label_idx = np.where(y == label)[0]
                    min_ = min(m, len(label_idx))
                    label_distance_matrix = distance_matrix[label_idx, :][:, label_idx]
                    kmedoids = KMedoids(n_clusters=min_, metric="precomputed").fit(
                        label_distance_matrix,
                    )
                    indices.extend(kmedoids.medoid_indices_[:m])
            elif method == "svc":
                svc = SVC(kernel="precomputed").fit(distance_matrix, y)
                support_idx = svc.support_
                summed_matrix = np.sum(distance_matrix, axis=0)
                sorted_idx = np.argsort(summed_matrix[support_idx])[
                    ::-1
                ]  # Sort in descending order
                indices.extend(sorted_idx[: m * n_classes])
            elif method == "random":
                keys = np.unique(y)
                values = [m] * len(keys)
                dict_ = dict(zip(keys, values))
                for label in np.unique(y):
                    label_idx = np.where(y == label)[0]
                    if len(label_idx) < m:
                        random_idx = np.random.choice(label_idx, m, replace=True)
                    else:
                        random_idx = np.random.choice(label_idx, m, replace=False)
                    indices.extend(random_idx)
            else:
                raise NotImplementedError(f"Method {method} not supported")
        elif method in ["hardness", "nearmiss", "knn"]:
            if method == "hardness":
                keys = np.unique(y)
                values = [m] * len(keys)
                dict_ = dict(zip(keys, values))
                model = InstanceHardnessThreshold(sampling_strategy=dict_)
            elif method == "nearmiss":
                keys = np.unique(y)
                values = [m] * len(keys)
                dict_ = dict(zip(keys, values))
                model = NearMiss(sampling_strategy=dict_)
            elif method == "knn":
                distance_matrix = pd.DataFrame(
                    distance_matrix,
                    columns=range(len(distance_matrix)),
                )
                y = pd.DataFrame(y, columns=["y"])
                y.index = list(range(len(y)))
                model = CondensedNearestNeighbour(sampling_strategy="not majority")
            else:
                raise NotImplementedError(f"Method {method} not supported")
            distance_matrix = pd.DataFrame(
                distance_matrix,
                columns=list(range(len(distance_matrix))),
            )
            distance_matrix, y = model.fit_resample(distance_matrix, y)
            y = pd.DataFrame(y, columns=["y"])
            y.index = list(range(len(y)))
            indices = y.index[: m * n_classes]
        else:
            raise NotImplementedError(f"Method {method} not supported")

        if len(indices) > len(self.X_):
            indices = indices[: len(self.X_)]
        return indices

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_jobs=-1,
        X_test=None,
        y_test=None,
        update=False,
    ):
        """Fit the model using X as training data and y as target values. If self.m is not -1, the best m samples will be selected using the method specified in self.condensing_method.

        Args:
            X (np.ndarray): The input data
            y (np.ndarray): The target labels

        Returns:
            GzipClassifier: The fitted model
        """
        assert len(X) == len(y), f"Expected {len(X)} == {len(y)}"
        logger.debug(f"Fitting with X of shape {X.shape} and y of shape {y.shape}")
        self.X_ = np.array(X) if not isinstance(X, np.ndarray) else X
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        if len(np.squeeze(y).shape) == 1:
            encoder = LabelBinarizer()
            self.y_ = encoder.fit_transform(y)
            self.n_classes_ = len(unique_labels(y))
            flat_y = np.squeeze(y).astype(int)
        else:
            self.y_ = y
            self.n_classes_ = y.shape[1]
            flat_y = np.argmax(y, axis=1)
        counts = np.bincount(flat_y)
        self.counts_ = counts
        logger.debug(f"Num Classes: {self.n_classes_}, counts: {counts}")
        self.n_features_ = X.shape[1] if len(X.shape) > 1 else 1
        self.classes_ = range(len(unique_labels(y)))

        if self.metric in compressors.keys():
            compressor = compressors[self.metric]
            Cx_ = Parallel(n_jobs=n_jobs)(delayed(compressor)(x) for x in self.X_)
            self.Cx_ = np.array(Cx_) if not isinstance(Cx_, np.ndarray) else Cx_
        else:
            self.Cx_ = None
            self.X_ = self.X_.astype(str)
        if self.m == 1 or self.m == -1:
            self.distance_matrix = self._prepare_training_matrix(n_jobs=n_jobs)
            self.distance_matrix = self.distance_matrix
        elif self.m > 0:
            assert isinstance(
                self.m,
                (int, float),
            ), f"Expected {self.m} to be an integer"
            assert isinstance(
                self.condensing_method,
                (str),
            ), f"Expected {self.condensing_method} to be a string"
            indices = self._find_best_samples(self.condensing_method)
            self._set_best_indices(indices)
        else:
            raise ValueError(
                f"Expected {self.m} to be -1, 0, a positive integer or a float between 0 and 1. Got type {type(self.m)}",
            )
        self.distance_matrix = self._prepare_training_matrix(n_jobs=n_jobs)
        self.clf_ = self.clf_.fit(self.distance_matrix, self.y_)
        return self

    def _set_best_indices(self, indices):
        self.X_ = self.X_[indices]
        self.y_ = self.y_[indices]
        if self.Cx_ is not None:
            self.Cx_ = self.Cx_[indices]
            # This is a hack that allows us to deal with n-dimensional arrays using the normal matrix[:, indices][indices, :] breaks if n>2
        distance_matrix = self.distance_matrix[
            indices
        ].T  # select the rows at the indices and transpose the matrix
        distance_matrix = distance_matrix[
            indices
        ]  # select the transposed columns at the indices
        self.distance_matrix = distance_matrix.T  # transpose the matrix again
        logger.debug(
            f"Selected {len(self.X_)} samples using method {self.condensing_method}.",
        )
        assert len(self.X_) == len(
            self.y_,
        ), f"Expected {len(self.X_)} == {len(self.y_)}"
        assert distance_matrix.shape == (
            len(self.X_),
            len(self.X_),
        ), f"Expected {distance_matrix.shape} == ({len(self.X_)}, {len(self.X_)})"

    def predict(self, X: np.ndarray):
        """Predict the class labels for the provided data.

        Args:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted class labels
        """
        check_is_fitted(self)
        logger.debug(f"Predicting with X of shape {X.shape}")
        if self.metric in compressors.keys():
            compressor = compressors[self.metric]
            Cx2 = Parallel(n_jobs=-1)(
                delayed(compressor)(x)
                for x in tqdm(
                    X,
                    desc="Compressing samples",
                    leave=False,
                    dynamic_ncols=True,
                )
            )
            assert len(Cx2) == len(X), f"Expected {len(Cx2)} == {len(X)}"
            assert len(self.X_) == len(
                self.Cx_,
            ), f"Expected {len(self.X_)} == {len(self.Cx_)}"
            distance_matrix = self._calculate_rectangular_distance_matrix(
                x1=X,
                Cx1=Cx2,
                x2=self.X_,
                Cx2=self.Cx_,
                n_jobs=-1,
            )
        else:
            distance_matrix = self._calculate_rectangular_distance_matrix(
                x2=self.X_,
                x1=X,
                n_jobs=-1,
            )
        assert distance_matrix.shape == (
            len(X),
            len(self.X_),
        ), f"Expected {distance_matrix.shape} == ({len(X)}, {len(self.X_)})"
        y_pred = self.clf_.predict(distance_matrix)

        if len(np.squeeze(y_pred).shape) == 1:
            encoder = LabelBinarizer()
            y_pred = encoder.fit(self.y_).transform(y_pred)
        else:
            encoder = LabelEncoder()
            y_pred = encoder.fit(self.y_).transform(y_pred)
        return y_pred

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


class BatchedGzipClassifier(BatchedMixin, GzipClassifier):

    pass


class GzipKNN(GzipClassifier):
    def __init__(
        self,
        k: int = 2,
        m=0,
        condensing_method="random",
        distance_matrix=None,
        metric="gzip",
        symmetric=False,
        similarity=False,
        double_centering=False,
        min_max_scale=False,
        modified=False,
        **kwargs,
    ):
        super().__init__(
            condensing_method=condensing_method,
            m=m,
            distance_matrix=distance_matrix,
            metric=metric,
            symmetric=symmetric,
            similarity=similarity,
            double_centering=double_centering,
            min_max_scale=min_max_scale,
            modified=modified,
            **kwargs,
        )
        self.clf_ = KNeighborsClassifier(n_neighbors=k, metric="precomputed", **kwargs)
        self.k = k

    def predict(self, X: np.ndarray, n_jobs=-1):
        """Predict the class labels for the provided data.

        Args:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted class labels
        """
        check_is_fitted(self)

        logger.debug(f"Predicting with X of shape {X.shape}")
        # Pre-compress samples not working
        if self.metric in compressors.keys():
            compressor = compressors[self.metric]
            Cx2 = Parallel(n_jobs=n_jobs)(
                delayed(compressor)(x)
                for x in tqdm(
                    X,
                    desc="Compressing samples",
                    leave=False,
                    dynamic_ncols=True,
                )
            )
            assert len(Cx2) == len(X), f"Expected {len(Cx2)} == {len(X)}"
            assert len(self.X_) == len(
                self.Cx_,
            ), f"Expected {len(self.X_)} == {len(self.Cx_)}"
            distance_matrix = self._calculate_rectangular_distance_matrix(
                x1=X,
                Cx1=Cx2,
                x2=self.X_,
                Cx2=self.Cx_,
                n_jobs=n_jobs,
            )
        else:
            distance_matrix = self._calculate_rectangular_distance_matrix(
                X,
                self.X_,
                n_jobs=n_jobs,
            )
        assert distance_matrix.shape == (
            len(X),
            len(self.X_),
        ), f"Expected {distance_matrix.shape} == ({len(X)}, {len(self.X_)})"
        y_pred = self.clf_.predict(distance_matrix)
        return y_pred


class BatchedGzipKNN(BatchedMixin, GzipKNN):
    pass


class GzipLogisticRegressor(GzipClassifier):
    def __init__(
        self,
        m=0,
        condensing_method="random",
        distance_matrix=None,
        metric="gzip",
        symmetric=False,
        similarity=False,
        double_centering=False,
        min_max_scale=False,
        modified=False,
        **kwargs,
    ):
        clf = LogisticRegression(**kwargs)
        super().__init__(
            clf_=clf,
            condensing_method=condensing_method,
            m=m,
            distance_matrix=distance_matrix,
            metric=metric,
            symmetric=symmetric,
            similarity=similarity,
            double_centering=double_centering,
            min_max_scale=min_max_scale,
            modified=modified,
            **kwargs,
        )


class BatchedGzipLogisticRegressor(BatchedMixin, GzipLogisticRegressor):
    pass


class GzipSVC(GzipClassifier):
    def __init__(
        self,
        kernel="rbf",
        m=0,
        condensing_method="random",
        distance_matrix=None,
        metric="gzip",
        symmetric=False,
        similarity=False,
        double_centering=False,
        min_max_scale=False,
        modified=False,
        **kwargs,
    ):
        clf = SVC(kernel=kernel, **kwargs)
        super().__init__(
            clf_=clf,
            condensing_method=condensing_method,
            m=m,
            distance_matrix=distance_matrix,
            metric=metric,
            symmetric=symmetric,
            similarity=similarity,
            double_centering=double_centering,
            min_max_scale=min_max_scale,
            modified=modified,
            **kwargs,
        )
        self.kernel = kernel


class BatchedGzipSVC(GzipSVC, BatchedMixin):
    pass


supported_models = {
    "knn": GzipKNN,
    "logistic": GzipLogisticRegressor,
    "svc": GzipSVC,
}

batched_models = {
    "knn": BatchedGzipKNN,
    "logistic": BatchedGzipLogisticRegressor,
    "svc": BatchedGzipSVC,
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
    batched=False,
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
        batched (bool): If True, a batched model will be used. Default is False.
        **kwargs: Additional keyword arguments to pass to the GzipClassifier
    Returns:
        dict: A dictionary containing the accuracy, train_time, and pred_time
    """
    if batched is True:
        _ = kwargs.pop("batched", "")
        model = batched_models[model_type](**kwargs)
    else:
        model = supported_models[model_type](**kwargs)
    alias = model_scorers[model_type]
    scorer = scorers[alias]
    start = time.time()

    model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
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
    return X_train, X_test, y_train, y_test


def main(args: argparse.Namespace):
    """
    This is the main function that runs the GzipClassifier with the provided arguments.
    It will fetch the dataset, split it into training and testing sets.
    Then, it will train the model using the fit method and test it using the predict method.
    Args:
        args (argparse.Namespace): The command line arguments
    Usage:
        python python gzip_classifier.py --metric gzip  --m 10 --condensing_method svc  --dataset kdd_nsl k=3
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
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    test_model(X_train, X_test, y_train, y_test, **params)


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
    "--m",
    type=int,
    default=-1,
    help="The number of best samples to use. If -1, all samples will be used.",
)
parser.add_argument(
    "--condensing_method",
    type=str,
    default="random",
    help=f"The method used to select the best training samples. Choices are {all_condensers}",
)
parser.add_argument(
    "--distance_matrix",
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


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
