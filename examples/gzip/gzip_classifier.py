"""
This is a module toa be used as a reference for building other modules
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
import gzip
from tqdm import tqdm
from pathlib import Path

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
        self.k = k
        self.compressor = compressor
        self.m = m
        self._set_compressor()
        self.method = method
        pathExists = Path(distance_matrix).exists()
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
        if  isinstance(self.distance_matrix, np.ndarray) and not isinstance(self.distance_matrix, type(None)):
            return self.distance_matrix

        isString = isinstance(self.distance_matrix, str)
        pathExists = Path(self.distance_matrix).exists()
        if isString and pathExists:
            return np.load(self.distance_matrix, allow_pickle=True)

        if isinstance(self.distance_matrix, str) and not pathExists:
            distance_matrix = np.zeros((len(x), len(x)))
            for i, xi in tqdm(enumerate(x), desc="Calculating distance matrix...", leave=False, total=x):
                # using the self._ncd method to calculate the distance
                distance_matrix[i] = self._ncd(Cx[i], str(xi))
            Path(self.distance_matrix).parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.distance_matrix, X=distance_matrix)
            # all other cases return something. why doesn't this one?
        else:
            distance_matrix = np.zeros((len(x), len(x)))
            for i, xi in tqdm(enumerate(x), desc="Calculating distance matrix...", leave=False, total=len(x)):
                # using the self._ncd method to calculate the distance
                distance_matrix[i] = self._ncd(Cx[i], str(xi))
            return distance_matrix
            
    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

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
        if self.compressor in compressors:
            self._compress = compressors(self.compressor)
        else:
            raise NotImplementedError(
                f"Compressing with {self.compressor} not supported."
            )

