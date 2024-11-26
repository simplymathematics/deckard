from typing import Literal
import logging
import numpy as np
import pandas as pd
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


from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups, make_classification
from sklearn.preprocessing import LabelEncoder


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
        coef0=0,
        degree=0,
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
            assert self.degree in [2], "Degree must be 2 for quadratic form"
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


def load_data(dataset, **kwargs):
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
    if isinstance(X, pd.DataFrame):
        X = [str(x) for x in X.values]
    elif isinstance(X, (list, np.ndarray)):
        X = [str(x) for x in X]
    else:
        raise ValueError(f"Unknown type {type(X)}")
    X = np.array(X)
    if len(kwargs) > 0:
        X, _, y, _ = train_test_split(X, y, **kwargs)
    return X, y


# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import roc_curve, auc
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection import ParameterGrid


# model1 = LogisticRegression(max_iter=1000)
# model2 = KNeighborsClassifier()
# model3 = SVC(kernel="precomputed")


# logistic_params = {
#     "model__tol": [1e-4, 1e-3, 1e-2],
#     "model__C": [0.1, 1, 10, 100],
#     "model__penalty": ["l1", "l2"],
#     "model__solver" : ["saga"]
# }

# knn_params = {
#     "model__n_neighbors": [1, 3, 5, 7, 9],
#     "model__weights": ["uniform", "distance"],
# }

# svc_params = {
#     "model__C": [0.1, 1, 10, 100],
# }

# exp_form = {
#     "kernelizer__degree" : [1, 2],
#     "kernelizer__gamma": [.0001, .001, .01, .1, 1, 10, 100, 1000],
#     "kernelizer__coef0" : [0],
# }
# exp_neg_form = {
#     "kernelizer__degree" : [1, 2],
#     "kernelizer__gamma": [.0001, .001, .01, .1, 1, 10, 100, 1000],
#     "kernelizer__coef0" : [0],
# }
# poly_form = {
#     "kernelizer__degree" : [1, 2, 3],
#     "kernelizer__gamma": [.0001, .001, .01, .1, 1, 10, 100, 1000],
#     "kernelizer__coef0": [0, 1, 10, 100],
# }
# quadratic_form = {
#     "kernelizer__gamma": [1],
#     "kernelizer__coef0": [0, 1, 10, 100],
#     "kernelizer__degree" : [2]
# }
# rational_form = {
#     "kernelizer__gamma": [1],
#     "kernelizer__coef0": [0, 1, 10, 100],
#     "kernelizer__degree" : [1]
# }
# multiquadric_form = {
#     "kernelizer__coef0": [0, 1, 10, 100],
#     "kernelizer__degree" : [2],
#     "kernelizer__gamma": [1],
# }

# kernelizers = [
#     exp_form,
#     exp_neg_form,
#     poly_form,
#     quadratic_form,
#     rational_form,
#     multiquadric_form,
# ]
# kernelizer_grid = list(ParameterGrid(kernelizers))

# transformer = StringDistanceTransformer(metric="gzip", n_jobs=-1)
# kernelizer = DistanceMatrixKernelizer(form="exp", gamma=1, degree=2)

# svc_list = []
# knn_list = []
# logistic_list = []
# lists_in_order = [logistic_list, knn_list, svc_list]
# i = 0
# for model_params in [logistic_params, knn_params, svc_params]:
#     model_list = lists_in_order[i]
#     for kernelizer_params in kernelizer_grid:
#         new_dict = {**model_params, **kernelizer_params, }
#         # Ensure that all values are lists
#         for key in new_dict:
#             if not isinstance(new_dict[key], list):
#                 new_dict[key] = [new_dict[key]]
#         model_list.append(new_dict)
#     i += 1


# pipeline1 = Pipeline([
#     ("transformer", transformer),
#     ("kernelizer", kernelizer),
#     ("model", model1)
# ])
# pipeline2 = Pipeline([
#     ("transformer", transformer),
#     ("kernelizer", kernelizer),
#     ("model", model2)
# ])
# pipeline3 = Pipeline([
#     ("transformer", transformer),
#     ("kernelizer", kernelizer),
#     ("model", model3)
# ])


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
                name: tmp.StringDistanceTransformer
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
                name: tmp.DistanceMatrixKernelizer
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
            name: art.attacks.evasion.ProjectedGradientDescent
            eps: .01
            # eps_step : ${eval:'(.1)*${.eps}'}
            batch_size : ${data.sample.test_size}
            targeted : false
        attack_size : ${data.sample.test_size}
        method : evasion
    files:
        data_file: tmp
        data_type: pkl
        reports : tmp
        model_file : tmp
        model_type : pkl
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
