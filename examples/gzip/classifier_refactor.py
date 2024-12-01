from typing import Literal
import logging
import numpy as np
import brotli
import pickle
import gzip
import zstd
import lzma
import bz2
from Levenshtein import distance, ratio, hamming, jaro, jaro_winkler, seqratio
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin


logger = logging.getLogger(__name__)


def _gzip_len(x):
    return len(gzip.compress(str(x).encode()))


def _lzma_len(x):

    return len(lzma.compress(str(x).encode()))


def _bz2_len(x):

    return len(bz2.compress(str(x).encode()))


def _zstd_len(x):

    return len(zstd.compress(str(x).encode()))


def _pickle_len(x):

    return len(pickle.dumps(x))


def _brotli_len(x):
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
                (x, y, Cx_i, Cy_j, metric, sorting_hack, zero_hack, average_hack),
            )
    distances = Parallel(n_jobs=-1)(
        delayed(distance_helper)(*args)
        for args in tqdm(queue, total=n * m, desc="Calculating distances.", leave=False)
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
                (x, y, Cx_i, Cy_j, metric, sorting_hack, zero_hack, average_hack),
            )
    distances = Parallel(n_jobs=-1)(
        delayed(distance_helper)(*args)
        for args in tqdm(queue, total=n * m, desc="Calculating distances.", leave=False)
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
    assert mtx.shape == (n, m), f"Matrix shape is {mtx.shape} but should be {(n, m)}"

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
                (x, y, Cx_i, Cy_j, metric, sorting_hack, zero_hack, average_hack),
            )
    distances = Parallel(n_jobs=-1)(
        delayed(distance_helper)(*args)
        for args in tqdm(queue, total=n * m, desc="Calculating distances.", leave=False)
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
    assert mtx.shape == (n, m), f"Matrix shape is {mtx.shape} but should be {(n, m)}"
    return mtx


class StringDistanceTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        metric: str,
        algorithm: Literal[None, "assume", "sort", "average"] = None,
        n_jobs: int = -1,
        zero_hack: bool = False,
        lower_triangle=False,
        upper_triangle=False,
    ):
        assert metric in all_metrics, f"Unknown metric {metric}"
        self.metric = metric
        self.zero_hack = zero_hack
        assert algorithm in [
            None,
            "assume",
            "sort",
            "average",
        ], f"Unknown algorithm {algorithm}"
        self.algorithm = algorithm
        if self.algorithm is None:
            self.sort_hack = False
            self.average_hack = False
        elif self.algorithm == "sort":
            self.sort_hack = True
            self.average_hack = False
        elif self.algorithm == "average":
            self.sort_hack = False
            self.average_hack = True
        else:
            raise ValueError(f"Unknown algorithm {algorithm}")
        assert (
            lower_triangle + upper_triangle < 2
        ), "Only one of lower_triangle and upper_triangle can be used"
        self.upper_triangle = upper_triangle
        self.lower_triangle = lower_triangle

        self.calculate_distance_matrix = calculate_rectangular_distance_matrix
        self.n_jobs = n_jobs

    def _save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def fit(self, X, y=None):
        X = np.array([str(x) for x in X])
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

    def transform(self, X, y=None):
        X = np.array([str(x) for x in X])
        mtx = self.calculate_distance_matrix(
            X,
            self.X_,
            metric=self.metric,
            sorting_hack=self.sort_hack,
            zero_hack=self.zero_hack,
            average_hack=self.average_hack,
        )
        return mtx

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {
            "metric": self.metric,
            "algorithm": self.algorithm,
            "n_jobs": self.n_jobs,
            "zero_hack": self.zero_hack,
            "lower_triangle": self.lower_triangle,
            "upper_triangle": self.upper_triangle,
        }

    def set_params(self, **params):
        for param in params:
            setattr(self, param, params[param])
        return self


class DistanceMatrixKernelizer(BaseEstimator, TransformerMixin):
    # From https://pdfs.semanticscholar.org/a9ee/f3769fe3686591a88cc831f9f685632f1b95.pdf
    def __init__(
        self,
        coef0=None,
        degree=None,
        gamma=1,
        form: Literal[
            "exp",
            "exp_neg",
            "poly",
            "quadratic",
            "rational",
            "multiquadric",
        ] = None,
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
                logger.warning(f"Degree must be 2 for {form} form. Setting degree to 2")
            self.degree = 2
        else:
            self.degree = degree

    def fit(self, X, y=None):
        if self.form == "exp":
            assert self.coef0 == 0, "coef0 must be 0 for exp form"
            self.kernel_function = lambda x: np.exp(x**self.degree / self.gamma)
        elif self.form == "exp_neg":
            assert self.coef0 == 0, "coef0 must be 0 for exp_neg form"
            self.kernel_function = lambda x: np.exp(-(x**self.degree) / self.gamma)
        elif self.form == "poly":
            self.kernel_function = (
                lambda x: (self.gamma * x + self.coef0) ** self.degree
            )
        elif self.form == "quadratic":
            assert self.degree == 2, "Degree must be 2 for quadratic form"
            assert self.gamma == 1, "Gamma must be 1 for quadratic form"
            self.kernel_function = lambda x: (x + self.coef0) ** self.degree
        elif self.form == "rational":
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

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":

    _config = """
    data:
        name: raw_data/ddos_undersampled_10000.csv
        target: 'Label'
        drop:
        - 'Timestamp' # Drop the timestamp column
        - 'Unnamed: 0' # Drop the index column
        sample:
            random_state : 0
            train_size : 100
            test_size : 100
            stratify: True
            shuffle : True
            n_splits: 5
            fold: -1
        alias: ddos
        sklearn_pipeline:
            label_binarizer:
                name: sklearn.preprocessing.LabelBinarizer
                y: True
            transformer:
                name: classifier_refactor.StringDistanceTransformer
                metric : gzip
                algorithm: sort
                n_jobs: -1
                lower_triangle: False
                zero_hack: False
        _target_: deckard.Data
    model:
        data  : ${data}
        init:
            name: sklearn.svm.SVC
            max_iter: 1000
            C: 1
            probability: True
        sklearn_pipeline:
            kernelizer:
                name: classifier_refactor.DistanceMatrixKernelizer
                coef0: 0
                degree: 2
                gamma: 1
                form: exp
        _target_: deckard.base.model.Model
        art:
            _target_ : deckard.base.model.art_pipeline.ArtPipeline
            library : sklearn
            initialize:
    attack:
        data: ${data}
        model: ${model}
        _target_ : deckard.base.attack.Attack
        init:
            model: ${model}
            _target_: deckard.base.attack.AttackInitializer
            name: art.attacks.evasion.HopSkipJump
            batch_size : ${data.sample.test_size}
            targeted : false
            max_iter : 100
            max_eval : 100
            init_eval : 10
        attack_size : ${data.sample.test_size}
        method : evasion
    files:
        data_file: tmp
        data_type: .pkl
        reports : tmp
        model_dir : models
        model_file : tmp
        model_type : .pkl
        directory: tmp
        reports: reports
        score_dict_file: score_dict.json
    scorers:
        accuracy:
            name : sklearn.metrics.accuracy_score
            alias: accuracy
            direction: maximize
        precision:
            name : sklearn.metrics.precision_score
            average: weighted
            alias: precision
            direction: maximize
        recall:
            name : sklearn.metrics.recall_score
            average: weighted
            alias: recall
            direction: maximize
        f1:
            name : sklearn.metrics.f1_score
            average: weighted
            alias: f1
            direction: maximize
    metrics:
        - train_time
        - predict_time
        - accuracy
        - precision
        - recall
        - f1
        - adv_success
        - adv_precision
        - adv_recall
        - adv_f1
        - adv_accuracy
        - adv_fit_time
    optimisers:
        - accuracy
        - adv_accuracy
    _target_: deckard.Experiment
    """

    from hydra.utils import instantiate
    import yaml
    from hashlib import md5

    _config = yaml.safe_load(_config)
    _config["files"]["name"] = md5(str(_config).encode("utf-8")).hexdigest()
    exp = instantiate(_config)
    score_dict = exp()
    print(score_dict)
