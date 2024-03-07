import logging
from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Union
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)
from ..utils import my_hash

__all__ = ["SklearnDataSampler"]


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
        logger.info(f"Calling SklearnDataSampler with params {asdict(self)}")
        params = deepcopy(asdict(self))
        stratify = params.pop("stratify", False)
        if stratify is True:
            stratify = y
        else:
            stratify = None
        test_size = params.pop("test_size")
        train_size = params.pop("train_size")
        time_series = params.pop("time_series")
        if time_series is not True:
            # if train_size + test_size == 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                train_size=train_size,
                stratify=stratify,
                **params,
            )
        else:
            if isinstance(train_size, float):
                train_size = int(train_size * len(X))
            if test_size is None:
                test_size = len(X) - train_size
            elif isinstance(test_size, float):
                test_size = int(test_size * len(X))
            if isinstance(train_size, type(None)):
                assert test_size is not None
                train_size = len(X) - test_size
            X_train = X[:train_size]
            X_test = X[train_size : train_size + test_size]  # noqa E203
            y_train = y[:train_size]
            y_test = y[train_size : train_size + test_size]  # noqa E203

        return [X_train, X_test, y_train, y_test]

    def __hash__(self):
        return int(my_hash(self), 16)
