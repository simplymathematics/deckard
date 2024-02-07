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
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd


logger = logging.getLogger(__name__)


# it makes sense to implement these outside the class
# since none of the functions actually use 'self'
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
    "pickle": _pickle_compressor,
}

class GzipClassifier(ClassifierMixin, BaseEstimator):
    """An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    def __init__(self, k=3, m=-1, compressor="gzip", method="random", distance_matrix=None):
        """
        Initialize the GzipClassifier object.

        Args:
            k (int): The value of k for k-nearest neighbors. Default is 3. 
            m (int): The value of m for m-nearest neighbors. Default is -1, which indicates using all training samples.
            compressor (str): The name of the compressor. Default is "gzip".
            method (str): The method used for classification. Default is "random".
            distance_matrix (str or np.ndarray): The path to a numpy file or a numpy array representing the distance matrix.
                If a path is provided, the file will be loaded. If an array is provided, it will be used directly.
                Default is None.

        Raises:
            ValueError: If distance_matrix is not a path to a numpy file or a numpy array.

        """
        logger.info(f"Initializing GzipClassifier with k={k}, m={m}, compressor={compressor}, method={method}, distance_matrix={distance_matrix}")
        self.k = k
        self.compressor = compressor
        self.m = m
        self._set_compressor()
        self.method = method
        if distance_matrix is not None: # Added this line because the next fails when it is None
            pathExists = Path(distance_matrix).exists()
        else: # Added this to handle the case when distance_matrix is None
            pathExists = False
        isString = isinstance(distance_matrix, str)
        if isString and pathExists:
            self.distance_matrix = np.load(distance_matrix, allow_pickle=True)['X']
        elif isString and not pathExists:
            self.distance_matrix = distance_matrix
        elif isinstance(distance_matrix, np.ndarray):
            self.distance_matrix = distance_matrix
        elif isinstance(distance_matrix, type(None)):
            self.distance_matrix = None
        else:
            raise ValueError(f"distance_matrix must be a path to a numpy file or a numpy array, got {type(distance_matrix)}")

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_features_ = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else 1
        self.X_ = np.array(X)
        self.y_ = np.array(y)
        Cxs = []
        logger.info(f"Training with {len(self.X_)} samples")
        # Convert all strings to gzip compressed strings
        for x in tqdm(self.X_, desc="Compressing...", leave=False):
            Cx = self._compress(x)
            Cxs.append(Cx)
        Cxs = np.array(Cxs)
        self.Cx_ = Cxs
        if self.m != -1 :
            # Calculate a distance matrix
            # For each class, find the m-nearest neighbors
            indices = self._find_best_training_samples(method = self.method)
            self.X_ = self.X_[indices]
            self.y_ = self.y_[indices]
            self.Cx_ = self.Cx_[indices]
            assert len(self.X_) == len(self.y_) == len(self.Cx_), f"Expected {len(self.X_)} == {len(self.y_)} == {len(self.Cx_)}"

    def _find_best_training_samples(self, method = "medoid"):
        """
        Args:
            method (str): The method used to select the best training samples. Default is "medoid". Choices are "sum", "mean", "medoid", "random", "knn", "svc".
        Returns:
            list: The indices of the best training samples.
        """
        distance_matrix = self._calculate_distance_matrix(self.X_, self.Cx_)
        indices = []
        if method == "sum":
            for label in self.classes_:
                label_idx = np.where(self.y_ == label)[0]
                label_distance_matrix = distance_matrix[label_idx, :]
                summed_matrix = np.sum(label_distance_matrix, axis=0)
                sorted_idx = np.argsort(summed_matrix)
                indices.extend(sorted_idx[: self.m])
        elif method == "mean":
            for label in self.classes_:
                label_idx = np.where(self.y_ == label)[0]
                label_distance_matrix = distance_matrix[label_idx, :]
                mean_matrix = np.mean(label_distance_matrix, axis=0)
                sorted_idx = np.argsort(mean_matrix)
                indices.extend(sorted_idx[: self.m])
        elif method == "medoid":
            from sklearn_extra.cluster import KMedoids
            for label in self.classes_:
                kmedoids = KMedoids(n_clusters=self.m, metric="precomputed").fit(distance_matrix)
                indices.extend(kmedoids.medoid_indices_)
        elif method == "random":
            for label in self.classes_:
                label_idx = np.where(self.y_ == label)[0]
                random_idx = np.random.choice(label_idx, self.m)
                indices.extend(random_idx)
        elif method == "knn":
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=self.m).fit(distance_matrix)
            _, indices = nn.kneighbors(distance_matrix, n_neighbors=self.m, return_distance=True)
            # TODO: Sort by distances
            # TODO: select m entries from each class
            indices = list(indices[: self.m * len(self.classes_)])
        elif method == "svc":
            from sklearn.svm import SVC
            svc = SVC(kernel="rbf").fit(distance_matrix, self.y_)
            indices.extend(svc.support_[: self.m * len(self.classes_)])
        else:
            raise NotImplementedError(f"Method {method} not supported")
        return indices
    
    
    
    # misleading name. this is considerably more than ncd
    # which makes it harder to optimize
    # +1
    def _ncd(self, Cx1, x1):
        distance_from_x1 = []
        for x2, Cx2 in zip(self.X_, self.Cx_):
            x2 = str(x2)
            x1x2 = " ".join([x1, x2])
            Cx1x2 = self._compress(x1x2)
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        return distance_from_x1

    def _calculate_distance_matrix(self, x, Cx):
        """
        Args:
            x (np.ndarray): The input data
            Cx (np.ndarray): The compressed input data
        Returns:
            np.ndarray: The distance matrix of size (len(x), len(x))
        """
        if  isinstance(self.distance_matrix, np.ndarray) and not isinstance(self.distance_matrix, type(None)):
            return self.distance_matrix

        isString = isinstance(self.distance_matrix, str)
        pathExists = Path(self.distance_matrix).exists()
        if isString and pathExists:
            return np.load(self.distance_matrix, allow_pickle=True)
        elif isinstance(self.distance_matrix, str) and not Path(self.distance_matrix).exists():
            pbar = tqdm(total=len(x), desc="Calculating distance matrix...", leave=False)
            logger.info(f"Calculating distance matrix and saving to {self.distance_matrix}")
            distance_matrix = np.zeros((len(x), len(x)))
            for i, xi in enumerate(x):
                # using the self._ncd method to calculate the distance
                distance_matrix[i] = self._ncd(Cx[i], str(xi))
                pbar.update(1)
            pbar.close()
            Path(self.distance_matrix).parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.distance_matrix, X=distance_matrix)
            # all other cases return something. why doesn't this one?
        else:
            distance_matrix = np.zeros((len(x), len(x)))
            pbar = tqdm(total=len(x), desc="Calculating distance matrix...", leave=False)
            for i, xi in tqdm(enumerate(x), desc="Calculating distance matrix...", leave=False, total=len(x)):
                # using the self._ncd method to calculate the distance
                distance_matrix[i] = self._ncd(Cx[i], str(xi))
                pbar.update(1)
            pbar.close()
            return distance_matrix
            
    def predict(self, X):
        
        """A scikit-learn implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_", "Cx_", "_compress"])

        # Input validation
        # X = check_array(X)
        results = []
        for x1 in tqdm(X, desc="Predicting...", leave=False, position=0, total=len(X)):
            x1 = str(x1)
            Cx1 = self._compress(x1)
            distance_from_x1 = []
            for x2, Cx2 in zip(self.X_, self.Cx_):
                x2 = str(x2)
                x1x2 = " ".join([x1, x2])
                Cx1x2 = self._compress(x1x2)
                ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
                distance_from_x1.append(ncd)
            distance_from_x1 = self._ncd(Cx1, x1)
            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_class = list(self.y_[sorted_idx[: self.k]])
            # predict class
            predict_class = max(set(top_k_class), key=top_k_class.count)
            results.append(predict_class)
        return results
    
    # A switch statement might be nicer than this
    # but those are only supported in python3.10 or later:
    # https://www.freecodecamp.org/news/python-switch-statement-switch-case-example/
    def _set_compressor(self):
        """ Selects the compressor based on the model initialization."""
        if self.compressor in compressors:
            self._compress = compressors[self.compressor]
        else:
            raise NotImplementedError(
                f"Compressing with {self.compressor} not supported."
            )


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
    print(f"Training time: {end - start}")
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
    parser.add_argument("--compressor", type=str, default="gzip")
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
