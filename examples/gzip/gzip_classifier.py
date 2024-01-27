"""
This is a module to be used as a reference for building other modules
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

# from sklearn.metrics import euclidean_distances, accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_classification, fetch_20newsgroups
import gzip
from tqdm import tqdm


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

    def __init__(self, k=3):
        self.k = k
        self.compressor = "gzip"
        self._set_compressor()

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
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]

        self.X_ = X
        self.y_ = y
        Cxs = []

        for x in self.X_:
            Cx = self._compress(x)
            Cxs.append(Cx)
        self.Cx_ = Cxs
        # Return the classifier
        return self

    def _set_compressor(self):
        if self.compressor == "gzip":
            self._compress = self._gzip_compressor
        else:
            raise NotImplementedError(
                f"Compressing with {self.compressor} not supported.",
            )

    def _gzip_compressor(self, x):
        return len(gzip.compress(str(x).encode()))

    def _ncd(self, Cx1, x1):
        distance_from_x1 = []
        for x2, Cx2 in zip(self.X_, self.Cx_):
            x2 = str(x2)
            x1x2 = " ".join([x1, x2])
            Cx1x2 = self._compress(x1x2)
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)

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
        for x1 in tqdm(X, desc="Predicting..."):
            x1 = str(x1)
            Cx1 = self._compress(x1)
            distance_from_x1 = []
            for x2, Cx2 in zip(self.X_, self.Cx_):
                x2 = str(x2)
                x1x2 = " ".join([x1, x2])
                Cx1x2 = self._compress(x1x2)
                ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
                distance_from_x1.append(ncd)
            # distance_from_x1 = self._ncd(Cx1, x1)
            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_class = list(self.y_[sorted_idx[: self.k]])
            predict_class = max(set(top_k_class), key=top_k_class.count)
            results.append(predict_class)
        return results
