import logging
from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from numpy import unique
from ..utils import my_hash

logger = logging.getLogger(__name__)

__all__ = ["SklearnDataSampler", "SklearnDataStratifiedSampler", "SklearnSplitSampler"]


@dataclass
class SklearnDataSampler:
    test_size: Union[float, int] = 0.2
    train_size: Union[float, int] = 0.8
    random_state: int = 0
    shuffle: bool = True
    stratify: bool = False
    time_series: bool = False

    def __init__(
        self,
        test_size=0.2,
        train_size=0.8,
        random_state=0,
        shuffle=True,
        stratify=False,
        time_series=False,
    ):
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.time_series = time_series

    def __call__(self, X, y):
        logger.debug(f"Calling SklearnDataSampler with params {asdict(self)}")
        params = deepcopy(asdict(self))
        stratify = params.pop("stratify", False)
        if stratify is True:
            y = LabelBinarizer().fit_transform(y)
            stratify = y
        else:
            stratify = None
        time_series = params.pop("time_series")
        params.pop("train_size")
        params.pop("test_size")
        # Ensure that the train_size and test_size are valid
        if time_series is not True:
            random_state = params.pop("random_state", 0)
            # split the data into train and test sets according to the maximum sizes possible (using the requested values as a minimum)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=self.train_size,
                test_size=self.test_size,
                stratify=stratify,
                random_state=random_state,
                **params,
            )
        else:
            if isinstance(self.train_size, float):
                self.train_size = int(self.train_size * len(X))
            if self.test_size is None:
                self.test_size = len(X) - self.train_size
            elif isinstance(self.test_size, float):
                self.test_size = int(self.test_size * len(X))
            if isinstance(self.train_size, type(None)):
                assert self.test_size is not None
                self.train_size = len(X) - self.test_size
            assert self.train_size + self.test_size <= len(
                X,
            ), "self.train_size + self.test_size must be <= len(X)"
            X_train = X[: self.train_size]
            X_test = X[self.train_size : self.train_size + self.test_size]  # noqa E203
            y_train = y[: self.train_size]
            y_test = y[self.train_size : self.train_size + self.test_size]  # noqa E203

        return [X_train, X_test, y_train, y_test]

    def __hash__(self):
        return int(my_hash(self), 16)

    def _determine_maximum_split(self, X, train_int, test_int):

        assert train_int + test_int <= len(
            X,
        ), "train_size + test_size must be == len(X)"
        max_train = len(X) - test_int
        max_test = len(X) - train_int
        if max_train + max_test > len(X):
            n_samples = len(X)
            remainder = n_samples - (max_train + max_test)
            max_train += remainder // 2
            max_test += remainder // 2
        return max_train, max_test


@dataclass
class SklearnDataStratifiedSampler:
    n_splits: int = 5
    random_state: int = 0
    shuffle: bool = True
    fold: int = 0

    def __init__(self, n_splits=5, random_state=0, shuffle=True, fold: int = 0):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        assert fold <= n_splits - 1, "fold must be less than n_splits - 1 (0-indexed)."
        assert fold >= -1, "fold must be greater than or equal to 0."
        self.fold = fold

    def __call__(self, X, y):
        stratifier = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        list_of_splits = list(stratifier.split(X, y))
        train_idx, test_idx = list_of_splits[self.fold]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        return [X_train, X_test, y_train, y_test]

    def __hash__(self):
        return int(my_hash(self), 16)


@dataclass
class SklearnSplitSampler:
    train_size: Union[float, int] = 0.8
    test_size: Union[float, int] = 0.2
    random_state: int = 0
    shuffle: bool = True
    stratify: bool = False
    n_splits: int = 1
    fold: int = -1
    time_series: bool = False

    def __init__(
        self,
        train_size=0.8,
        test_size=0.2,
        random_state=0,
        shuffle=True,
        stratify=False,
        n_splits=1,
        fold=-1,
        time_series=False,
    ):
        assert fold <= n_splits - 1, "fold must be less than n_splits - 1 (0-indexed)."
        self.train_size = train_size
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.n_splits = n_splits
        self.fold = fold
        self.time_series = time_series

    def __call__(self, X, y):
        logger.debug(f"Calling SklearnDataSampler with params {asdict(self)}")
        params = deepcopy(asdict(self))
        params.pop("n_splits", 1)
        params.pop("fold", -1)
        sampler = SklearnDataSampler(**params)
        X_train_big, X_eval, y_train_big, y_eval = sampler(X, y)
        logger.info(
            f"X_train_big.shape: {X_train_big.shape}, X_eval.shape: {X_eval.shape}",
        )
        logger.info(
            f"y_train_big.shape: {y_train_big.shape}, y_eval.shape: {y_eval.shape}",
        )
        if self.fold == -1 or self.n_splits == 1:
            res = [X_train_big, X_eval, y_train_big, y_eval]
        else:
            # Make sure that y_train_big, y_eval are 2D arrays
            binarizer = LabelBinarizer()
            y_train_big = binarizer.fit_transform(y_train_big)
            y_eval = binarizer.transform(y_eval)
            u_train, c_train = unique(y_train_big, return_counts=True)
            u_eval, c_eval = unique(y_eval, return_counts=True)
            for u, c in zip(u_train, c_train):
                logger.debug(f"Train class {u} has {c} samples")
            for u, c in zip(u_eval, c_eval):
                logger.debug(f"Eval class {u} has {c} samples")
            stratified_sampler = SklearnDataStratifiedSampler(
                n_splits=self.n_splits,
                random_state=self.random_state,
                shuffle=self.shuffle,
                fold=self.fold,
            )
            X_train, X_test, y_train, y_test = stratified_sampler(
                X_train_big,
                y_train_big,
            )
            res = [X_train, X_test, y_train, y_test]
        logger.debug(f"X_train.shape: {res[0].shape}, X_test.shape: {res[1].shape}")
        logger.debug(f"y_train.shape: {res[2].shape}, y_test.shape: {res[3].shape}")
        return res

    def __hash__(self):
        return int(my_hash(self), 16)
