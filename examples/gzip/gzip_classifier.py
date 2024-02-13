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
# python -m pip install numpy scikit-learn tqdm scikit-learn-extra pandas

import numpy as np
import gzip
from tqdm import tqdm
from pathlib import Path
import logging
import time
import argparse
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn_extra.cluster import KMedoids
            
import pandas as pd
from joblib import Parallel, delayed
from typing import Literal


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

def ncd(x1, x2, compressor:Literal["gzip", "lzma", "bz2", "zstd", "pkl", None]="gzip") -> float:
    """
    Calculate the normalized compression distance between two objects treated as strings.
    Args:
        x1 (str): The first object
        x2 (str): The second object
    Returns:
        float: The normalized compression distance between x1 and x2
    """
    compressors = {
        "gzip": _gzip_compressor,
        "lzma": _lzma_compressor,
        "bz2": _bz2_compressor,
        "zstd": _zstd_compressor,
        "pkl": _pickle_compressor,
    }
    compressor = compressors[compressor]
    x1 = str(x1)
    x2 = str(x2)
    Cx1 = compressor(x1)
    Cx2 = compressor(x2) 
    x1x2 = " ".join([x1, x2])
    Cx1x2 = len(gzip.compress(x1x2.encode()))
    min_ = min(Cx1, Cx2)
    max_ = max(Cx1, Cx2)
    ncd = (Cx1x2 - min_) / max_
    return ncd



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
        The name of the compressor to use. Choices are "gzip", "lzma", "bz2", "zstd", "pkl", "pickle".
    method: str, default="random"
        The method used to select the best training samples. Choices are "sum", "mean", "medoid", "random", "knn", "svc".
    metric: str, default="ncd"
        The metric used to calculate the distance between samples. Choices are "ncd".
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

    def __init__(self, k=3, m=-1, compressor="gzip", method="random", distance_matrix=None, metric='ncd', symmetric=True):
        """
        Initialize the GzipClassifier object.

        Args:
            k (int): The value of k for k-nearest neighbors. Default is 3. 
            m (int): The value of m for  m-best samples. Default is -1, which indicates using all training samples.
            compressor (str): The name of the compressor. Default is "gzip".
            method (str): The method used for classification. Default is "random".
            metric (str): The metric used to calculate the distance between samples. Default is "ncd".
            distance_matrix (str or np.ndarray): The path to a numpy file or a numpy array representing the distance matrix.
                If a path is provided, the file will be loaded. If an array is provided, it will be used directly.
                Default is None.

        Raises:
            ValueError: If distance_matrix is not a path to a numpy file or a numpy array.
            NotImplementedError: If the metric is not supported.
        """
        logger.info(f"Initializing GzipClassifier with k={k}, m={m}, compressor={compressor}, method={method}, distance_matrix={distance_matrix}")
        self.k = k
        self.m = m
        self.method = method
        if  isinstance(distance_matrix, str) and Path(distance_matrix).exists():
            self.distance_matrix = np.load(distance_matrix, allow_pickle=True)['X']
        elif isinstance(distance_matrix, str) and not Path(distance_matrix).exists():
            self.distance_matrix = distance_matrix
        elif isinstance(distance_matrix, np.ndarray):
            self.distance_matrix = distance_matrix
        elif isinstance(distance_matrix, type(None)):
            self.distance_matrix = None
        else:
            raise ValueError(f"distance_matrix must be a path to a numpy file or a numpy array, got {type(distance_matrix)}")
        self.metric = metric
        if self.metric == "ncd":
            self._distance = ncd
            self.compressor= compressor
        else:
            raise NotImplementedError(f"Metric {self.metric} not supported")
        self.symmetric = symmetric
    
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        """Fit the model using X as training data and y as target values. If self.m is not -1, the best m samples will be selected using the method specified in self.method.

        Args:
            X (np.ndarray): The input data
            y (np.ndarray): The target labels

        Returns:
            GzipClassifier: The fitted model
        """
        assert len(X) == len(y), f"Expected {len(X)} == {len(y)}"
        logger.info(f"Fitting GzipClassifier with X of shape {X.shape} and y of shape {y.shape}")
        self.X_ = np.array(X) if not isinstance(X, np.ndarray) else X
        self.y_ = np.array(y) if not isinstance(y, np.ndarray) else y
        self.n_classes_ = len(unique_labels(y)) if len(unique_labels(y)) > 2 else 2
        self.n_features_ = X.shape[1]
        self.classes_ = unique_labels(y)
        if self.m != -1:
            indices = self._find_best_samples(self.method)
            self.X_ = self.X_[indices]
            self.y_ = self.y_[indices]
        assert len(self.X_) == len(self.y_), f"Expected {len(self.X_)} == {len(self.y_)}"
        return self

    def predict(self, X:np.ndarray):
        """Predict the class labels for the provided data.

        Args:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted class labels
        """
        check_is_fitted(self)
        
        y_pred = []
        distance_matrix = np.zeros((len(X), len(self.X_)))
        # Iterate over the samples provided by th euser
        for i in tqdm(range(len(X)), desc="Predicting", leave=False, position=0, total=len(X)):
            # Iterate over the training samples
            distance_matrix[i,:] = Parallel(n_jobs=-1)(delayed(self._distance)(X[i], self.X_[j], compressor=self.compressor) for j in range(len(self.X_)))
            # Sort the distances and get the nearest k samples
            sorted_idx = np.argsort(distance_matrix[i])
            # Get the labels of the nearest samples
            nearest_labels = list(self.y_[sorted_idx[: self.k]])
            # predict class
            unique, counts = np.unique(nearest_labels, return_counts=True)
            # Get the most frequent label
            y_pred.append(unique[np.argmax(counts)])
        return y_pred

    def _find_best_samples(self, method = "medoid"):
        """
        Args:
            method (str): The method used to select the best training samples. Default is "medoid". Choices are "sum", "mean", "medoid", "random", "knn", "svc".
        Returns:
            list: The indices of the best training samples.
        """
        distance_matrix = self._prepare_distance_matrix()
        indices = []
        if method == "sum":
            for label in range(self.n_classes_):
                label_idx = np.where(self.y_ == label)[0]
                label_distance_matrix = distance_matrix[label_idx, :]
                summed_matrix = np.sum(label_distance_matrix, axis=0)
                sorted_idx = np.argsort(summed_matrix)
                indices.extend(sorted_idx[: self.m])
        elif method == "mean":
            for label in range(self.n_classes_):
                label_idx = np.where(self.y_ == label)[0]
                label_distance_matrix = distance_matrix[label_idx, :]
                mean_matrix = np.mean(label_distance_matrix, axis=0)
                sorted_idx = np.argsort(mean_matrix)
                indices.extend(sorted_idx[: self.m])
        elif method == "medoid":
            for label in range(self.n_classes_):
                kmedoids = KMedoids(n_clusters=self.m, metric="precomputed").fit(distance_matrix)
                indices.extend(kmedoids.medoid_indices_)
        elif method == "random":
            for label in range(self.n_classes_):
                label_idx = np.where(self.y_ == label)[0]
                random_idx = np.random.choice(label_idx, self.m)
                indices.extend(random_idx)
        elif method == "knn":
            nn = NearestNeighbors(n_neighbors=self.m).fit(distance_matrix)
            _, indices = nn.kneighbors(distance_matrix, n_neighbors=self.m, return_distance=True)
            # TODO: Sort by distances
            # TODO: select m entries from each class
            indices = list(indices[: self.m * self.n_classes_])
        elif method == "svc":
            svc = SVC(kernel="precomputed").fit(distance_matrix, self.y_)
            # TODO: Sort by distances
            # TODO: select m entries from each class
            indices.extend(svc.support_[: self.m * self.n_classes_])
        else:
            raise NotImplementedError(f"Method {method} not supported")
        return indices
    
    def _calculate_square_distance_matrix(self, x1, x2, n_jobs=-1):
        """
        Calculate the distance matrix between two sets of objects, treating them as strings, assuming d(a,b) != d(b,a)
        Args:
            x1 (np.ndarray): The first set of objects
            x2 (np.ndarray): The second set of objects
        Returns:
            np.ndarray: The distance matrix of size (len(x1), len(x2))
        """
        matrix_ = np.zeros((len(x1), len(x2)))
        pbar = tqdm(total=len(x1), desc="Calculating triangular distance matrix")
        for i in range(len(x1)):
            # Parallelize the calculation of the distance matrix
            matrix_[i, :] = Parallel(n_jobs=n_jobs)(delayed(self._distance)(x1[i], x2[j], compressor=self.compressor) for j in range(len(x2)))
            pbar.update(1)
        pbar.close()
        assert matrix_.shape == (len(x1), len(x2)), f"Expected {matrix_.shape} == ({len(x1)}, {len(x2)})"
        return matrix_
    
    
    def _calculate_triangular_distance_matrix(self, x1, x2, n_jobs=-1):
        """
        Calculate the distance matrix between two sets of objects, treating them as strings. Assuming the d(a,b) = d(b,a)
        Args:
            x1 (np.ndarray): The first set of objects
            x2 (np.ndarray): The second set of objects
        Returns:
            np.ndarray: The distance matrix of size (len(x1), len(x2))
        """
        matrix_ = np.zeros((len(x1), len(x2)))
        pbar = tqdm(total=len(x1), desc="Calculating triangular distance matrix")
        for i in range(len(x1)):
            # Parallelize the calculation of the distance matrix
            matrix_[i, :i] = Parallel(n_jobs=n_jobs)(delayed(self._distance)(x1[i], x2[j], compressor=self.compressor) for j in range(i))
            # Copy the lower triangular part to the upper triangular part
            matrix_[i, :i] = matrix_[:i, i]
            pbar.update(1)
        pbar.close()
        assert matrix_.shape == (len(x1), len(x2)), f"Expected {matrix_.shape} == ({len(x1)}, {len(x2)})"
        return matrix_
    
    def _load_distance_matrix(self, path):
        if Path(path).exists():
            return np.load(path, allow_pickle=True)['X']
        else:
            raise FileNotFoundError(f"Distance matrix file {path} not found")
    
    def _save_distance_matrix(self, path, matrix):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=matrix)
    
    def _prepare_distance_matrix(self):
        """
        Prepare the distance matrix for classification. 
        If self.distance_matrix is a path to a numpy file, it will be loaded.
        If it is a numpy array, it will be used directly. 
        If it is None, the distance matrix will be calculated using self.X_ and self.X_.
        """
        if self.symmetric is True:
            self._calculate_distance_matrix = self._calculate_triangular_distance_matrix
        else:
            self._calculate_distance_matrix = self._calculate_square_distance_matrix
        if isinstance(self.distance_matrix, str) and Path(self.distance_matrix).exists():
            distance_matrix = self._load_distance_matrix(self.distance_matrix)
        elif isinstance(self.distance_matrix, str) and not Path(self.distance_matrix).exists():
            distance_matrix = self._calculate_distance_matrix(self.X_, self.X_)
            self._save_distance_matrix(self.distance_matrix, distance_matrix)
        elif isinstance(self.distance_matrix, np.ndarray):
            distance_matrix = self.distance_matrix
        elif isinstance(self.distance_matrix, type(None)):
            distance_matrix = self._calculate_distance_matrix(self.X_, self.X_)
        else:
            raise ValueError(f"distance_matrix must be a path to a numpy file or a numpy array, got {type(self.distance_matrix)}")
        return distance_matrix
    
    
    

def test_model(X, y, train_size = 100, test_size =100, **kwargs) -> dict:
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
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size, test_size=test_size, stratify=y, random_state=42)
    model = GzipClassifier(**kwargs)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    train_time = end - start
    start = time.time()
    predictions = model.predict(X_test)
    end = time.time()
    pred_time = end - start
    score = round(accuracy_score(y_test, predictions), 3)
    print(f"Training time: {train_time}")
    print(f"Prediction time: {pred_time}")
    print(f"Accuracy score is: {score}")
    return {
        "accuracy": score,
        "train_time": train_time,
        "pred_time": pred_time,
    }


def main(args:argparse.Namespace):
    """
    This is the main function that runs the GzipClassifier with the provided arguments. 
    It will fetch the dataset, split it into training and testing sets.
    Then, it will train the model using the fit method and test it using the predict method.
    Args:
        args (argparse.Namespace): The command line arguments
    Usage:
        python gzip_classifier.py --compressor gzip --k 3 --m 100 --method random --distance_matrix distance_matrix --dataset kdd_nsl
    """
    if args.dataset == "20newsgroups":
        X, y = fetch_20newsgroups(subset='train', categories=["alt.atheism", "talk.religion.misc"], shuffle=True, random_state=42, return_X_y=True)
        y = LabelEncoder().fit(y).transform(y) # Turns the labels "alt.atheism" and "talk.religion.misc" into 0 and 1
    elif args.dataset == "kdd_nsl":
        df = pd.read_csv("https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/34e546e4813f154d11d4f13869b9e3481fc3e829/kdd-nsl.csv", header=None)
        width = df.shape[1]
        y = df[width-2] # the 2nd to last column is the target
        del df[width-2] # remove the target from the dataframe
        X = np.array(df)
        del df
        new_y = []
        for entry in y: # convert the target to binary from 'normal' and various attacks.
            if entry == "normal":
                new_y.append(0)
            else:
                new_y.append(1)
        y = LabelEncoder().fit(new_y).transform(new_y)
    elif args.dataset == "make_classification":
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        y = LabelEncoder().fit(y).transform(y)
    elif args.dataset == "truthseeker":
        df = pd.read_csv("https://gist.githubusercontent.com/simplymathematics/8c6c04bd151950d5ea9e62825db97fdd/raw/34e546e4813f154d11d4f13869b9e3481fc3e829/truthseeker.csv")
        y = np.array(df['BotScoreBinary'].astype("int"))
        del df['BotScoreBinary']
        del df['BotScore']
        del df['statement']
        X = np.array(df)
    else:
        raise ValueError(f"Dataset {args.dataset} not found")
    params = vars(args)
    params.pop("dataset")
    test_model(X, y, train_size=args.train_size, test_size=args.test_size, k=args.k, m=args.m, method=args.method, distance_matrix=args.distance_matrix, compressor=args.compressor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--not_symmetric", dest="symmetric", action="store_false")
    parser.add_argument("--compressor", type=str, default="gzip")
    parser.add_argument("--metric", type=str, default="ncd")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--m", type=int, default=-1)
    parser.add_argument("--method", type=str, default="random")
    parser.add_argument("--distance_matrix", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="kdd_nsl")
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
