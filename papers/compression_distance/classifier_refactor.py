from typing import Literal
import logging
import numpy as np
import pickle
import gzip
import lzma
import bz2
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)

try:
    import brotli  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency
    brotli = None

try:
    import zstd  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency
    zstd = None

try:
    from Levenshtein import (  # type: ignore[reportMissingImports]
        distance,
        ratio,
        hamming,
        jaro,
        jaro_winkler,
        seqratio,
    )
except ImportError:  # pragma: no cover - optional dependency
    distance = ratio = hamming = jaro = jaro_winkler = seqratio = None


def _require_levenshtein(fn, name: str):
    if fn is None:
        raise ImportError(
            f"Levenshtein is not installed, required for metric '{name}'",
        )
    return fn


def _gzip_len(x):
    return len(gzip.compress(str(x).encode()))


def _lzma_len(x):

    return len(lzma.compress(str(x).encode()))


def _bz2_len(x):

    return len(bz2.compress(str(x).encode()))


def _zstd_len(x):
    if zstd is None:
        raise ImportError("zstd is not installed")
    return len(zstd.compress(str(x).encode()))


def _pickle_len(x):

    return len(pickle.dumps(x))


def _brotli_len(x):
    if brotli is None:
        raise ImportError("brotli is not installed")
    return len(brotli.compress(str(x).encode()))


compressors = {
    "gzip": _gzip_len,
    "lzma": _lzma_len,
    "bz2": _bz2_len,
    "zstd": _zstd_len,
    "pkl": _pickle_len,
    "brotli": _brotli_len,
}


string_metrics = {
    "levenshtein": lambda x, y: _require_levenshtein(distance, "levenshtein")(
        x,
        y,
    ),
    "ratio": lambda x, y: _require_levenshtein(ratio, "ratio")(x, y),
    "hamming": lambda x, y: _require_levenshtein(hamming, "hamming")(
        x,
        y,
    ),
    "jaro": lambda x, y: _require_levenshtein(jaro, "jaro")(x, y),
    "jaro_winkler": lambda x, y: _require_levenshtein(
        jaro_winkler,
        "jaro_winkler",
    )(x, y),
    "seqratio": lambda x, y: _require_levenshtein(seqratio, "seqratio")(
        x,
        y,
    ),
}

all_metrics = {
    **compressors,
    **string_metrics,
}


def ncd(x, y, Cx=None, Cy=None, metric="gzip"):
    if Cx is None:
        Cx = compressors[metric](x)
    if Cy is None:
        Cy = compressors[metric](y)
    Cxy = compressors[metric](x + y)
    return (Cxy - min(Cx, Cy)) / max(Cx, Cy)


def string_distance(x, y, metric="levenshtein"):
    return string_metrics[metric](x, y)


def calculate_distance(x, y, Cx=None, Cy=None, metric="gzip"):
    if metric in compressors:
        return ncd(x, y, Cx, Cy, metric=metric)
    elif metric in string_metrics:
        return string_distance(x, y, metric=metric)
    else:
        raise ValueError(f"Unknown metric {metric}")


def distance_helper(
    x,
    y,
    Cx=None,
    Cy=None,
    metric="gzip",
    sorting_hack=False,
    zero_hack=False,
    average_hack=False,
):
    if zero_hack:
        if x == y:
            return 0
    assert (
        sorting_hack + average_hack < 2
    ), "Only one of sorting_hack and average_hack can be used"
    if sorting_hack:
        lenx = len(x)
        leny = len(y)
        if lenx >= leny:
            x1 = x
            y1 = y
            Cx1 = Cx
            Cy1 = Cy
        else:
            x1 = y
            y1 = x
            Cx1 = Cy
            Cy1 = Cx
        x = x1
        y = y1
        Cx = Cx1
        Cy = Cy1
        result = calculate_distance(x, y, Cx, Cy, metric=metric)
    elif average_hack:
        dist1 = calculate_distance(x, y, Cx, Cy, metric=metric)
        dist2 = calculate_distance(y, x, Cy, Cx, metric=metric)
        result = (dist1 + dist2) / 2
    else:
        result = calculate_distance(x, y, Cx, Cy, metric=metric)
    return result


def calculate_rectangular_distance_matrix(
    X,
    Y,
    metric="gzip",
    sorting_hack=False,
    zero_hack=False,
    average_hack=False,
):
    n = len(X)
    m = len(Y)
    if metric in compressors:
        Cx = [compressors[metric](x) for x in X]
        Cy = [compressors[metric](y) for y in Y]
        assert len(Cx) == n, "Cx must have the same length as X"
        assert len(Cy) == m, "Cy must have the same length as Y"
    else:
        Cx = [None] * n
        Cy = [None] * m
    queue = []
    for i in range(n):
        for j in range(m):
            x = X[i]
            y = Y[j]
            Cx_i = Cx[i]
            Cy_j = Cy[j]
            queue.append(
                (
                    x,
                    y,
                    Cx_i,
                    Cy_j,
                    metric,
                    sorting_hack,
                    zero_hack,
                    average_hack,
                ),
            )
    distances = Parallel(n_jobs=-1, prefer="threads")(
        delayed(distance_helper)(*args)
        for args in tqdm(
            queue,
            total=n * m,
            desc="Calculating distances.",
            leave=False,
        )
    )
    # Reformat the distances into a matrix
    distances = np.array(distances).reshape(n, m)
    return distances


def calculate_lower_triangular_distance_matrix(
    X,
    Y,
    metric="gzip",
    sorting_hack=False,
    zero_hack=False,
    average_hack=False,
):
    n = len(X)
    m = len(Y)
    assert m == n, "Lower triangular matrix can only be calculated for square matrices"
    if metric in compressors:
        Cx = [compressors[metric](x) for x in X]
        Cy = [compressors[metric](y) for y in Y]
    else:
        Cx = [None] * n
        Cy = [None] * m
    queue = []
    for i in range(n):
        for j in range(i + 1, m):
            x = X[i]
            y = Y[j]
            Cx_i = Cx[i]
            Cy_j = Cy[j]
            queue.append(
                (
                    x,
                    y,
                    Cx_i,
                    Cy_j,
                    metric,
                    sorting_hack,
                    zero_hack,
                    average_hack,
                ),
            )
    distances = Parallel(n_jobs=-1, prefer="threads")(
        delayed(distance_helper)(*args)
        for args in tqdm(
            queue,
            total=n * m,
            desc="Calculating distances.",
            leave=False,
        )
    )
    # get lower triangular indices
    indices = np.tril_indices(n)
    # Reformat the distances into a matrix
    mtx = np.zeros((n, m))
    mtx[indices] = distances
    old_diag = np.diag(np.diag(mtx))
    mtx = mtx + mtx.T - old_diag
    new_diag = np.diag(np.diag(mtx))
    assert np.all(new_diag == old_diag), "Diagonal elements have changed"
    assert mtx.shape == (
        n,
        m,
    ), f"Matrix shape is {mtx.shape} but should be {(n, m)}"

    return mtx


def calculate_upper_triangular_distance_matrix(
    X,
    Y,
    metric="gzip",
    sorting_hack=False,
    zero_hack=False,
    average_hack=False,
):
    n = len(X)
    m = len(Y)
    assert m == n, "Upper triangular matrix can only be calculated for square matrices"
    if metric in compressors:
        Cx = [compressors[metric](x) for x in X]
        Cy = [compressors[metric](y) for y in Y]
    else:
        Cx = [None] * n
        Cy = [None] * m
    queue = []
    for i in range(n):
        for j in range(i, m):
            x = X[i]
            y = Y[j]
            Cx_i = Cx[i]
            Cy_j = Cy[j]
            queue.append(
                (
                    x,
                    y,
                    Cx_i,
                    Cy_j,
                    metric,
                    sorting_hack,
                    zero_hack,
                    average_hack,
                ),
            )
    distances = Parallel(n_jobs=-1, prefer="threads")(
        delayed(distance_helper)(*args)
        for args in tqdm(
            queue,
            total=n * m,
            desc="Calculating distances.",
            leave=False,
        )
    )
    # Reformat the distances into a matrix
    mtx = np.zeros((n, m))
    indices = np.triu_indices(n)
    mtx[indices] = distances
    old_diag = np.diag(np.diag(mtx))
    # Flip the matrix to get the lower triangular part, then add the two matrices together
    # Subtract the double counted diagonal elements
    mtx = mtx + mtx.T - old_diag
    new_diag = np.diag(np.diag(mtx))
    assert np.all(new_diag == old_diag), "Diagonal elements have changed"
    assert mtx.shape == (
        n,
        m,
    ), f"Matrix shape is {mtx.shape} but should be {(n, m)}"
    return mtx


class StringDistanceTransformer(BaseEstimator, TransformerMixin):

    @staticmethod
    def _coerce_samples(X):
        """Convert tabular or array-like input into a 1D array of sample strings."""
        if hasattr(X, "itertuples"):
            return np.asarray(
                [" ".join(map(str, row)) for row in X.itertuples(index=False, name=None)],
                dtype=object,
            )
        if hasattr(X, "tolist") and hasattr(X, "shape") and len(getattr(X, "shape", ())) == 2:
            rows = X.tolist()
            return np.asarray([" ".join(map(str, row)) for row in rows], dtype=object)
        return np.asarray([str(x) for x in X], dtype=object)

    def __init__(
        self,
        metric: str,
        algorithm: Literal[None, "assume", "sort", "average"] = None,
        n_jobs: int = -1,
        zero_hack: bool = False,
        sort_hack: bool = False,
        average_hack: bool = False,
        lower_triangle=False,
        upper_triangle=False,
        distance_matrix_full: str | None = None,
        distance_matrix_train: str | None = None,
        distance_matrix_test: str | None = None,
        train_indices: list | None = None,
        test_indices: list | None = None,
    ):
        assert metric in all_metrics, f"Unknown metric {metric}"
        self.metric = metric
        self.zero_hack = zero_hack
        self.algorithm = algorithm
        self.sort_hack = sort_hack
        self.average_hack = average_hack
        assert (
            lower_triangle + upper_triangle < 2
        ), "Only one of lower_triangle and upper_triangle can be used"
        self.upper_triangle = upper_triangle
        self.lower_triangle = lower_triangle
        self.distance_matrix_full = distance_matrix_full
        self.distance_matrix_train = distance_matrix_train
        self.distance_matrix_test = distance_matrix_test
        self.train_indices = train_indices
        self.test_indices = test_indices
        self._full_matrix = None

        self.calculate_distance_matrix = calculate_rectangular_distance_matrix
        self.n_jobs = n_jobs

    def _save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def set_split_indices(self, train_indices=None, test_indices=None, val_indices=None):
        if train_indices is not None:
            self.train_indices = list(train_indices)
        if test_indices is not None:
            self.test_indices = list(test_indices)
        return self

    def _load_matrix_file(self, path):
        matrix = np.load(path)
        if isinstance(matrix, np.lib.npyio.NpzFile):
            keys = matrix.files
            if "data" in keys:
                arr = matrix["data"]
            elif len(keys) > 0:
                arr = matrix[keys[0]]
            else:
                matrix.close()
                raise ValueError(f"No arrays found in matrix file {path}")
            matrix.close()
            return arr
        return matrix

    def _save_matrix_file(self, path, matrix):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if path_obj.suffix == ".npz":
            np.savez_compressed(path_obj, data=matrix)
        else:
            np.save(path_obj, matrix)

    def _slice_from_full_matrix(self, full_matrix, row_indices, col_indices):
        row_idx = np.asarray(row_indices, dtype=int)
        col_idx = np.asarray(col_indices, dtype=int)
        return full_matrix[np.ix_(row_idx, col_idx)]

    def pre_sample_fit(self, X, y=None, data=None):
        if not self.distance_matrix_full:
            return self
        X = self._coerce_samples(X)
        expected_n = len(X)
        try:
            loaded_matrix = self._load_matrix_file(self.distance_matrix_full)
            if loaded_matrix.shape == (expected_n, expected_n):
                self._full_matrix = loaded_matrix
                return self
        except (FileNotFoundError, OSError, ValueError):
            pass

        full_matrix = calculate_rectangular_distance_matrix(
            X,
            X,
            metric=self.metric,
            sorting_hack=self.sort_hack,
            zero_hack=self.zero_hack,
            average_hack=self.average_hack,
        )
        self._save_matrix_file(self.distance_matrix_full, full_matrix)
        self._full_matrix = full_matrix
        return self

    def fit(self, X, y=None):
        X = self._coerce_samples(X)
        if self.distance_matrix_full:
            if self._full_matrix is None:
                try:
                    self._full_matrix = self._load_matrix_file(self.distance_matrix_full)
                except (FileNotFoundError, OSError, ValueError):
                    self.pre_sample_fit(X, y=y)
            if self._full_matrix is not None:
                if self.train_indices is not None:
                    self.mtx_ = self._slice_from_full_matrix(
                        self._full_matrix,
                        self.train_indices,
                        self.train_indices,
                    )
                else:
                    self.mtx_ = self._full_matrix
                self.X_ = X
                return self

        # If pre-computed train matrix is provided, load it; otherwise calculate
        if self.distance_matrix_train:
            try:
                self.mtx_ = self._load_matrix_file(self.distance_matrix_train)
                self.X_ = X
                return self
            except (FileNotFoundError, OSError):
                pass  # Fall back to on-the-fly calculation
        
        if self.lower_triangle:
            self.calculate_fit_matrix = calculate_lower_triangular_distance_matrix
            self.lower_triangle = True
        elif self.upper_triangle:
            self.calculate_fit_matrix = calculate_upper_triangular_distance_matrix
            self.upper_triangle = True
        else:
            self.calculate_fit_matrix = calculate_rectangular_distance_matrix
        self.mtx_ = self.calculate_fit_matrix(
            X,
            X,
            metric=self.metric,
            sorting_hack=self.sort_hack,
            zero_hack=self.zero_hack,
            average_hack=self.average_hack,
        )
        self.X_ = X
        return self

    def transform(self, X, y=None):
        X = self._coerce_samples(X)
        if self.distance_matrix_full and self._full_matrix is not None:
            if len(X) == len(self.X_):
                return self.mtx_
            if self.test_indices is not None and self.train_indices is not None:
                return self._slice_from_full_matrix(
                    self._full_matrix,
                    self.test_indices,
                    self.train_indices,
                )

        # If pre-computed test matrix is provided and X is test data, load it
        if self.distance_matrix_test and len(X) != len(self.X_):
            try:
                return self._load_matrix_file(self.distance_matrix_test)
            except (FileNotFoundError, OSError):
                pass  # Fall back to on-the-fly calculation
        
        mtx = self.calculate_distance_matrix(
            X,
            self.X_,
            metric=self.metric,
            sorting_hack=self.sort_hack,
            zero_hack=self.zero_hack,
            average_hack=self.average_hack,
        )
        return mtx

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=y, **fit_params)
        return self.transform(X, y=y)

    def get_params(self, deep=True):
        return {
            "metric": self.metric,
            "algorithm": self.algorithm,
            "n_jobs": self.n_jobs,
            "zero_hack": self.zero_hack,
            "sort_hack": self.sort_hack,
            "average_hack": self.average_hack,
            "lower_triangle": self.lower_triangle,
            "upper_triangle": self.upper_triangle,
            "distance_matrix_full": self.distance_matrix_full,
            "distance_matrix_train": self.distance_matrix_train,
            "distance_matrix_test": self.distance_matrix_test,
            "train_indices": self.train_indices,
            "test_indices": self.test_indices,
        }

    def set_params(self, **params):
        for param in params:
            setattr(self, param, params[param])
        return self


class DistanceMatrixKernelizer(BaseEstimator, TransformerMixin):
    # From https://pdfs.semanticscholar.org/a9ee/f3769fe3686591a88cc831f9f685632f1b95.pdf
    def __init__(
        self,
        coef0: float = 0,
        degree: int | None = None,
        gamma=1,
        form: Literal[
            "exp",
            "exp_neg",
            "poly",
            "quadratic",
            "rational",
            "multiquadric",
        ] | None = None,
    ):
        self.coef0 = coef0
        self.gamma = gamma
        assert form in [
            "exp",
            "exp_neg",
            "poly",
            "quadratic",
            "rational",
            "multiquadric",
        ], f"Unknown form: {form}"
        self.form = form
        if self.form in ["multiquadric", "quadratic"]:
            if degree != 2:
                logger.warning(
                    f"Degree must be 2 for {form} form. Setting degree to 2",
                )
            self.degree = 2
        else:
            self.degree = degree

    def fit(self, X, y=None):
        if self.form == "exp":
            assert self.coef0 == 0, "coef0 must be 0 for exp form"
            if self.degree is None:
                raise ValueError("degree must be set for exp form")
            degree = self.degree
            self.kernel_function = lambda x: np.exp(x**degree / self.gamma)
        elif self.form == "exp_neg":
            assert self.coef0 == 0, "coef0 must be 0 for exp_neg form"
            if self.degree is None:
                raise ValueError("degree must be set for exp_neg form")
            degree = self.degree
            self.kernel_function = lambda x: np.exp(
                -(x**degree) / self.gamma,
            )
        elif self.form == "poly":
            if self.degree is None:
                raise ValueError("degree must be set for poly form")
            degree = self.degree
            self.kernel_function = (
                lambda x: (self.gamma * x + self.coef0) ** degree
            )
        elif self.form == "quadratic":
            assert self.degree == 2, "Degree must be 2 for quadratic form"
            assert self.gamma == 1, "Gamma must be 1 for quadratic form"
            self.kernel_function = lambda x: (x + self.coef0) ** self.degree
        elif self.form == "rational":
            if self.degree is None:
                raise ValueError("degree must be set for rational form")
            assert self.degree == 1, "Degree must be 1 for rational form"
            assert self.gamma == 1, "Gamma must be 1 for rational form"
            self.kernel_function = lambda x: 1 - (x) / (x + self.coef0)
        elif self.form == "multiquadric":
            assert self.degree == 2, "Degree must be 2 for multiquadric form"
            self.gamma = 1, "Gamma must be 1 for multiquadric form"
            self.kernel_function = lambda x: 1 / np.sqrt(x**2 + self.coef0**2)
        else:
            raise ValueError(f"Unknown form {self.form}")

    def transform(self, X, y=None):
        return self.kernel_function(X)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=y, **fit_params)
        return self.transform(X, y=y)


class KernelToDistanceTransformer(BaseEstimator, TransformerMixin):
    """Convert a similarity kernel matrix into a distance matrix."""

    def __init__(
        self,
        form: Literal[
            "exp",
            "exp_neg",
            "hamming",
            "rational",
            "poly",
            "quadratic",
            "multiquadric",
        ] = "exp_neg",
        normalize_unit_diagonal: bool = False,
        assume_unit_diagonal: bool | None = None,
    ):
        if form not in [
            "exp",
            "exp_neg",
            "hamming",
            "rational",
            "poly",
            "quadratic",
            "multiquadric",
        ]:
            raise ValueError(f"Unknown kernel form: {form}")
        if normalize_unit_diagonal and assume_unit_diagonal is False:
            raise ValueError(
                "normalize_unit_diagonal requires assume_unit_diagonal to be "
                "True or None",
            )
        self.form = form
        self.normalize_unit_diagonal = normalize_unit_diagonal
        self.assume_unit_diagonal = assume_unit_diagonal

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        kernel_matrix = np.asarray(X)
        if kernel_matrix.ndim != 2:
            raise ValueError(
                f"Kernel matrix must be 2D, got shape {kernel_matrix.shape}",
            )

        if self.normalize_unit_diagonal:
            if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
                if self.assume_unit_diagonal is not True:
                    raise ValueError(
                        "normalize_unit_diagonal requires a square kernel matrix "
                        "unless assume_unit_diagonal=True",
                    )
            else:
                diagonal = np.diag(kernel_matrix)
                if np.any(diagonal <= 0):
                    raise ValueError(
                        "Cannot normalize kernel matrix with non-positive diagonal entries",
                    )
                kernel_matrix = kernel_matrix / np.sqrt(
                    np.outer(diagonal, diagonal),
                )

        if self.assume_unit_diagonal is True or (
            self.assume_unit_diagonal is None
            and self.form in {"exp", "exp_neg", "hamming", "rational"}
        ):
            return 2 - 2 * kernel_matrix

        if kernel_matrix.shape[0] != kernel_matrix.shape[1]:
            raise ValueError(
                "Rectangular kernel matrices require a unit-diagonal kernel form "
                "or assume_unit_diagonal=True",
            )

        diagonal = np.diag(kernel_matrix)
        return diagonal[:, None] + diagonal[None, :] - 2 * kernel_matrix

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y=y)
