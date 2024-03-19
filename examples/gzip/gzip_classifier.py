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

from joblib import Parallel, delayed
from typing import Literal

from batchMixin import BatchedMixin

logger = logging.getLogger(__name__)


def _gzip_compressor(x):
    return len(gzip.compress(str(x).encode()))


def _lzma_compressor(x):
    import lzma

    return len(lzma.compress(str(x).encode()))


def _bz2_compressor(x):
    import bz2

    return len(bz2.compress(str(x).encode()))


def _zstd_compressor(x):
    import zstd

    return len(zstd.compress(str(x).encode()))


def _pickle_compressor(x):
    import pickle

    return len(pickle.dumps(x))


compressors = {
    "gzip": _gzip_compressor,
    "lzma": _lzma_compressor,
    "bz2": _bz2_compressor,
    "zstd": _zstd_compressor,
    "pkl": _pickle_compressor,
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

    compressor = (
        compressors[method] if method in compressors.keys() else compressors["gzip"]
    )
    x1 = str(x1)
    x2 = str(x2)
    Cx1 = compressor(x1) if cx1 is None else cx1
    Cx2 = compressor(x2) if cx2 is None else cx2
    x1x2 = " ".join([x1, x2])
    Cx1x2 = compressor(x1x2)
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


def _calculate_string_distance(x1, x2, method):
    x1 = str(x1)
    x2 = str(x2)
    if method in string_metrics.keys():
        dist = string_metrics[method]
    else:
        raise NotImplementedError(
            f"Method {method} not supported. Supported methods are: {string_metrics.keys()}",
        )
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
        sampling_method="random",
        distance_matrix=None,
        metric="gzip",
        symmetric=False,
        precompute=True,
        **kwargs,
    ):
        """
        Initialize the GzipClassifier object.

        Args:
            k (int): The value of k for k-nearest neighbors. Default is 3.
            m (int): The value of m for  m-best samples. Default is -1, which indicates using all training samples.
            sampling_method (str): The method used for classification. Default is "random".
            metric (str): The metric used to calculate the distance between samples. Default is "ncd".
            distance_matrix (str or np.ndarray): The path to a numpy file or a numpy array representing the distance matrix.
                If a path is provided, the file will be loaded. If an array is provided, it will be used directly.
                Default is None.
            symmetric (bool): If True, the distance matrix will be treated as symmetric. Default is False.
            precompute (bool): If True, the distance matrix will be precomputed and stored in self.distance_matrix during the fit method and a sklearn KNeighborsClassifier object will be created and stored in self.clf_.

        Raises:
            ValueError: If distance_matrix is not a path to a numpy file or a numpy array.
            NotImplementedError: If the metric is not supported.
        """
        kwarg_string = str([f"{key}={value}" for key, value in kwargs.items()])
        logger.info(
            f"Initializing GzipClassifier with  m={m},  method={sampling_method}, distance_matrix={distance_matrix}, metric={metric}, symmetric={symmetric}, precompute={precompute}, {kwarg_string}",
        )
        self.m = m
        self.sampling_method = sampling_method
        if metric in compressors.keys():
            logger.info(f"Using NCD metric with {metric} compressor.")
            self._distance = ncd
            self.metric = metric
        elif metric in string_metrics.keys():
            logger.info(f"Using {metric} metric")
            self._distance = _calculate_string_distance
            self.metric = metric
        else:
            raise NotImplementedError(
                f"Metric {metric} not supported. Supported metrics are: ncd, {string_metrics.keys()} and {compressors.keys()}",
            )

        self.symmetric = symmetric
        if self.symmetric is True:
            self._calculate_distance_matrix = (
                self._calculate_lower_triangular_distance_matrix
            )
        else:
            self._calculate_distance_matrix = (
                self._calculate_rectangular_distance_matrix
            )
        self.precompute = precompute  # If True, the distance matrix will be precomputed and stored in self.distance_matrix during the fit method and a sklearn KNeighborsClassifier object will be created and stored in self.clf_.
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
        pbar = tqdm(
            total=len(x1),
            desc="Calculating asymmetric distance matrix.",
            leave=False,
            dynamic_ncols=True,
        )
        Cx1 = Cx1 if Cx1 is not None else [None] * len(x1)
        Cx2 = Cx2 if Cx2 is not None else [None] * len(x2)
        for i in range(len(x1)):
            # Parallelize the calculation of the distance matrix
            if self.metric in compressors.keys():
                matrix_[i, :] = Parallel(n_jobs=n_jobs)(
                    delayed(self._distance)(
                        x1[i],
                        x2[j],
                        cx1=Cx1[i],
                        cx2=Cx2[j],
                        method=self.metric,
                    )
                    for j in range(len(x2))
                )
            else:
                matrix_[i, :] = Parallel(n_jobs=n_jobs)(
                    delayed(self._distance)(x1[i], x2[j], method=self.metric)
                    for j in range(len(x2))
                )
            pbar.update(1)
        pbar.close()
        assert matrix_.shape == (
            len(x1),
            len(x2),
        ), f"Expected {matrix_.shape} == ({len(x1)}, {len(x2)})"
        return matrix_

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

        matrix_ = np.zeros((len(x1), len(x2)))
        pbar = tqdm(
            total=len(x1),
            desc="Calculating symmetric distance metrix.",
            leave=False,
            dynamic_ncols=True,
        )
        Cx1 = Cx1 if Cx1 is not None else [None] * len(x1)
        Cx2 = Cx2 if Cx2 is not None else [None] * len(x2)
        for i in range(len(x1)):
            # Parallelize the calculation of the distance matrix
            if self.metric in compressors.keys():
                matrix_[i, :i] = Parallel(n_jobs=n_jobs)(
                    delayed(self._distance)(
                        x1[i],
                        x2[j],
                        cx1=Cx1[i],
                        cx2=Cx2[j],
                        method=self.metric,
                    )
                    for j in range(i)
                )
            else:
                matrix_[i, :i] = Parallel(n_jobs=n_jobs)(
                    delayed(self._distance)(x1[i], x2[j], method=self.metric)
                    for j in range(i)
                )
            # Copy the lower triangular part to the upper triangular part
            matrix_[i, :i] = matrix_[:i, i]
            pbar.update(1)
        pbar.close()
        assert matrix_.shape == (
            len(x1),
            len(x2),
        ), f"Expected {matrix_.shape} == ({len(x1)}, {len(x2)})"
        return matrix_

    def calculate_upper_triangular_distance_matrix(
        self,
        x1,
        x2,
        Cx1=None,
        Cx2=None,
        n_jobs=-1,
    ):
        matrix_ = np.zeros((len(x1), len(x2)))
        pbar = tqdm(
            total=len(x1),
            desc="Calculating symmetric distance metrix.",
            leave=False,
            dynamic_ncols=True,
        )
        Cx1 = Cx1 if Cx1 is not None else [None] * len(x1)
        Cx2 = Cx2 if Cx2 is not None else [None] * len(x2)
        for i in range(len(x1)):
            if self.metric in compressors.keys():
                matrix_[i, i:] = Parallel(n_jobs=n_jobs)(
                    delayed(self._distance)(
                        x1[i],
                        x2[j],
                        cx1=Cx1[i],
                        cx2=Cx2[j],
                        method=self.metric,
                    )
                    for j in range(i, len(x2))
                )
            else:
                matrix_[i, i:] = Parallel(n_jobs=n_jobs)(
                    delayed(self._distance)(x1[i], x2[j], method=self.metric)
                    for j in range(i, len(x2))
                )
            # copy the upper triangular part to the lower triangular part
            matrix_[i, i:] = matrix_[i:, i]
            pbar.update(1)
        pbar.close()
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

    def _prepare_training_matrix(self, n_jobs=-1):
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
        elif isinstance(self.distance_matrix, np.ndarray):
            distance_matrix = self.distance_matrix
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
        return distance_matrix

    def _find_best_samples(self, method="medoid", n_jobs=-1):
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
            y = pd.DataFrame(y, columns=["y"])
            y.index = list(range(len(y)))
            distance_matrix, y = model.fit_resample(distance_matrix, y)
            indices = y.index[: m * n_classes]
        else:
            raise NotImplementedError(f"Method {method} not supported")
        return indices

    def fit(self, X: np.ndarray, y: np.ndarray, n_jobs=-1):
        """Fit the model using X as training data and y as target values. If self.m is not -1, the best m samples will be selected using the method specified in self.sampling_method.

        Args:
            X (np.ndarray): The input data
            y (np.ndarray): The target labels

        Returns:
            GzipClassifier: The fitted model
        """
        assert len(X) == len(y), f"Expected {len(X)} == {len(y)}"
        logger.info(f"Fitting with X of shape {X.shape} and y of shape {y.shape}")
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
        logger.info(f"Num Classes: {self.n_classes_}, counts: {counts}")
        self.n_features_ = X.shape[1] if len(X.shape) > 1 else 1
        self.classes_ = range(len(unique_labels(y)))

        if self.metric in compressors.keys():
            compressor = compressors[self.metric]
            Cx_ = Parallel(n_jobs=n_jobs)(delayed(compressor)(x) for x in self.X_)
            self.Cx_ = np.array(Cx_) if not isinstance(Cx_, np.ndarray) else Cx_
        else:
            self.Cx_ = None
            self.X_ = self.X_.astype(str)
        if self.m > 0:
            assert isinstance(
                self.m,
                (int, float),
            ), f"Expected {self.m} to be an integer"
            assert isinstance(
                self.sampling_method,
                (str, type(None)),
            ), f"Expected {self.sampling_method} to be a string or None"
            indices = self._find_best_samples(self.sampling_method)
            self._set_best_indices(indices)
        elif self.m == -1:
            distance_matrix = self._prepare_training_matrix(n_jobs=n_jobs)
            self.distance_matrix = distance_matrix
        elif self.m is None or self.m == 0:
            pass
        else:
            raise ValueError(
                f"Expected {self.m} to be -1, 0, a positive integer or a float between 0 and 1. Got type {type(self.m)}",
            )
        if self.precompute is True:
            self.distance_matrix = self._prepare_training_matrix(n_jobs=n_jobs)
            self.clf_ = self.clf_.fit(self.distance_matrix, self.y_)
        else:
            raise NotImplementedError(
                f"Precompute {self.precompute} not supported for type(self.clf_) {type(self.clf_)}",
            )
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
        logger.info(
            f"Selected {len(self.X_)} samples using method {self.sampling_method}.",
        )
        counts = np.bincount(np.argmax(self.y_, axis=1))
        logger.info(f"Num Classes: {self.n_classes_}, counts: {counts}")
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
        logger.info(f"Predicting with X of shape {X.shape}")
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


class BatchedGzipClassifier(GzipClassifier, BatchedMixin):
    pass


class GzipKNN(GzipClassifier):
    def __init__(
        self,
        k: int = 2,
        m=0,
        sampling_method="random",
        distance_matrix=None,
        metric="gzip",
        symmetric=False,
        precompute=True,
        **kwargs,
    ):
        super().__init__(
            sampling_method=sampling_method,
            m=m,
            distance_matrix=distance_matrix,
            metric=metric,
            symmetric=symmetric,
            precompute=precompute,
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

        logger.info(f"Predicting with X of shape {X.shape}")
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
        y_pred = []
        if self.precompute is True:
            y_pred = self.clf_.predict(distance_matrix)
        else:
            for i in tqdm(
                range(len(X)),
                desc="Predicting",
                leave=False,
                total=len(X),
                dynamic_ncols=True,
            ):
                # Sort the distances and get the nearest k samples
                sorted_idx = np.argsort(distance_matrix[i])
                # Get the first k samples
                nearest_k = sorted_idx[: self.k]
                # Get the labels of the nearest samples
                nearest_labels = list(self.y_[nearest_k])
                # predict class
                unique, counts = np.unique(nearest_labels, return_counts=True)
                # Get the most frequent label
                y_pred.append(unique[np.argmax(counts)])
        return y_pred


class BatchedGzipKNN(GzipKNN, BatchedMixin):
    pass


class GzipLogisticRegressor(GzipClassifier):
    def __init__(
        self,
        m=0,
        sampling_method="random",
        distance_matrix=None,
        metric="gzip",
        symmetric=False,
        precompute=True,
        **kwargs,
    ):
        self.precompute = precompute
        clf = LogisticRegression(**kwargs)
        super().__init__(
            clf_=clf,
            precompute=precompute,
            sampling_method=sampling_method,
            m=m,
            distance_matrix=distance_matrix,
            metric=metric,
            symmetric=symmetric,
            **kwargs,
        )


class BatchedGzipLogisticRegressor(GzipLogisticRegressor, BatchedMixin):
    pass


class GzipSVC(GzipClassifier):
    def __init__(
        self,
        kernel="rbf",
        m=0,
        sampling_method="random",
        distance_matrix=None,
        metric="gzip",
        symmetric=False,
        precompute=True,
        **kwargs,
    ):
        self.precompute = precompute
        clf = SVC(kernel=kernel, **kwargs)
        super().__init__(
            clf_=clf,
            precompute=precompute,
            sampling_method=sampling_method,
            m=m,
            distance_matrix=distance_matrix,
            metric=metric,
            symmetric=symmetric,
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
        X (np.ndarray): The input data
        y (np.ndarray): The target labels
        train_size (int): The number of samples to use for training. Default is 100.
        test_size (int): The number of samples to use for testing. Default is 100.
        **kwargs: Additional keyword arguments to pass to the GzipClassifier
    Returns:
        dict: A dictionary containing the accuracy, train_time, and pred_time
    """
    if batched is True:
        model = batched_models[model_type](**kwargs)
    else:
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
    print(f"{alias}  is: {score}")
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
        df = pd.read_csv("raw_data/kdd_nsl.csv")
        y = df["label"]
        X = df.drop("label", axis=1)
    elif dataset == "kdd_nsl":
        df = pd.read_csv("raw_data/kdd_nsl.csv")
        y = df["label"]
        X = df.drop("label", axis=1)
        X = np.array(X)
    elif dataset == "make_classification":
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=2,
            random_state=42,
        )
        y = LabelEncoder().fit(y).transform(y)
    elif dataset == "truthseeker":
        df = pd.read_csv("raw_data/truthseeker.csv")
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
        python gzip_classifier.py --compressor gzip --k 3 --m 100 --method random --distance_matrix distance_matrix --dataset kdd_nsl
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
    params.update(**kwarg_args)
    params["precompute"] = True
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    test_model(X_train, X_test, y_train, y_test, **params)


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="knn")
parser.add_argument("--precompute", action="store_true")
parser.add_argument("--symmetric", action="store_true")
parser.add_argument("--metric", type=str, default="gzip", choices=all_metrics)
parser.add_argument("--m", type=int, default=-1)
parser.add_argument("--sampling_method", type=str, default="random")
parser.add_argument("--distance_matrix", type=str, default=None)
parser.add_argument("--dataset", type=str, default="kdd_nsl")
parser.add_argument("--train_size", type=int, default=100)
parser.add_argument("--test_size", type=int, default=100)
parser.add_argument("--optimizer", type=str, default="accuracy")
parser.add_argument("--precompressed", action="store_true")
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("kwargs", nargs=argparse.REMAINDER)

if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
