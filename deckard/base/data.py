import logging
import os
import pickle
from .hashable import my_hash
from pathlib import Path
from types import NoneType
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# mnist dataset from
from art.utils import load_dataset

logger = logging.getLogger(__name__)
from typing import Union

SUPPORTED_DATASETS = ["mnist", "iris", "cifar10"]
from deckard.base.hashable import BaseHashable


class Data(BaseHashable):
    """
    :attribute dataset: The dataset to use. Can be either a csv file, a string, or a pickled Data object.
    :attribute target: The target column to use. If None, the last column is used.
    :attribute time_series: If True, the dataset is treated as a time series. Default is False.
    :attribute train_size: The percentage of the dataset to use. Default is 0.1.
    :attribute random_state: The random state to use. Default is 0.
    :attribute shuffle: If True, the data is shuffled. Default is False.
    :attribute X_train: The training data. Created during initialization.
    :attribute X_test: The testing data. Created during initialization.
    :attribute y_train: The training target. Created during initialization.
    :attribute y_test: The testing target. Created during initialization.

    """

    def __init__(
        self,
        dataset: str,
        target=False,
        time_series: bool = False,
        train_size: int = 100,
        random_state=0,
        shuffle: bool = True,
        stratify=False,
        test_size: int = 100,
        path=None,
    ):
        """
        Initializes the data object.
        :param dataset: The dataset to use. Can be either a csv file, a string, or a pickled Data object.
        :param target: The target column to use. If None, the last column is used.
        :param time_series: If True, the dataset is treated as a time series. Default is False.
        :param train_size: The percentage of the dataset to use. Default is 0.1.
        :param random_state: The random state to use. Default is 0.
        :param train_size: The percentage of the dataset to use for testing. Default is 0.2.
        :param shuffle: If True, the data is shuffled. Default is False.
        :param flatten: If True, the dataset is flattened. Default is False.
        """
        self.dataset = dataset
        self.target = target
        self.time_series = time_series
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.test_size = test_size
        self.path = False
        tmp = dict(vars(self))
        self.params = tmp

    def __call__(self) -> None:
        if self.path is None or self.path == False:
            self.path = "."
        if isinstance(self.dataset, str) and not Path(self.path, self.dataset).exists():
            if self.dataset.endswith(".csv") or self.dataset.endswith(".txt"):
                (X_train, y_train), (X_test, y_test) = self._parse_csv(
                    self.dataset, self.target
                )
            elif self.dataset in SUPPORTED_DATASETS:
                (X_train, y_train), (X_test, y_test), _, _ = load_dataset(self.dataset)
            else:
                raise ValueError(
                    "String dataset must be a csv file or a supported dataset."
                )
            (
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                minimum,
                maximum,
            ) = self._sample_data(X_train, y_train, X_test, y_test)
            self.clip_values = (minimum, maximum)
        elif (
            isinstance(self.dataset, Union[str, Path])
            and Path(self.path, self.dataset).exists()
        ):
            with open(Path(self.path, self.dataset), "rb") as f:
                data = pickle.load(f)
            for key, value in vars(data).items():
                if hasattr(self.params, key):
                    setattr(self.params, key, value)
                setattr(self, key, value)
        else:
            raise ValueError(
                f"Dataset must be a csv file, a string, or a pickled Data object. Strings must be a supported dataset: {SUPPORTED_DATASETS}"
            )
        assert hasattr(self, "X_train"), "X_train is not defined. Something went wrong."
        assert hasattr(self, "X_test"), "X_test is not defined. Something went wrong."
        assert hasattr(self, "y_train"), "y_train is not defined. Something went wrong."
        assert hasattr(self, "y_test"), "y_test is not defined. Something went wrong."
        assert hasattr(
            self, "clip_values"
        ), "clip_values is not defined. Something went wrong."

    def set_params(self, **kwargs):
        """
        :param params: A dictionary of parameters to set.
        Sets the parameters of the data object.
        """
        self.__init__(**kwargs)
        try:
            self.__call__()
        except Exception as e:
            raise

    def _sample_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """
        :param dataset: A string specifying the dataset to use. Supports mnist, iris, stl10, cifar10, nursery, diabetes, and an arbitrary csv.
        :param target: The target column to use. If None, the last column is used.
        Chooses the dataset to use. Returns self.
        """
        # sets target, if specified
        if self.target is not None:
            self.target = self.target
        big_X = np.vstack((X_train, X_test))
        big_y = np.vstack((y_train, y_test))
        # sets stratify to None if stratify is False
        stratify = big_y if self.stratify == True else None
        assert len(big_X) == len(
            big_y
        ), "length of X is: {}. length of y is: {}".format(len(big_X), len(big_y))
        assert big_X.shape[0] == big_y.shape[0], "X has {} rows. y has {} rows.".format(
            big_X.shape[0], big_y.shape[0]
        )
        if self.train_size == 1:
            pass
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                big_X,
                big_y,
                train_size=self.train_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify,
            )
        if hasattr(self, "test_size"):
            assert isinstance(self.test_size, int) or isinstance(
                self.test_size, NoneType
            ), "test_size must be an integer"
            if self.test_size is not None and self.test_size <= len(X_test):
                stratify = y_test if self.stratify == True else None
                _, X_test, _, y_test = train_test_split(
                    X_test,
                    y_test,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    shuffle=self.shuffle,
                    stratify=stratify,
                )
                X_test = X_test[: self.test_size]
                y_test = y_test[: self.test_size]
                assert (
                    len(X_test) == self.test_size
                ), "X_test has {} rows. test_size is {}".format(
                    len(X_test), self.test_size
                )
            else:
                logger.warning(
                    "test_size is greater than the number of test samples. Setting test_size to {}".format(
                        len(X_test)
                    )
                )
                pass
        else:
            pass
        maximum = np.max(X_train)
        minimum = np.min(X_train)
        return X_train, y_train, X_test, y_test, minimum, maximum

    def _parse_csv(self, dataset: str = "mnist", target=None) -> None:
        """
        :param dataset: A string specifying the dataset to use. Supports mnist, iris, stl10, cifar10, nursery, and diabetes
        Chooses the dataset to use. Returns self.
        """
        assert dataset.endswith(".csv") or dataset.endswith(
            ".txt"
        ), "Dataset must be a .csv  or .txt file"
        df = pd.read_csv(dataset)
        df = df.dropna(axis=0, how="any")
        if target is None:
            self.target = df.columns[-1]
        else:
            self.target = target
        y = df[self.target]
        X = df.drop(self.target, axis=1)
        if self.time_series == False:
            stratify = y if self.stratify == True else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=self.train_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify,
            )
            maximum = max(X_train)
            minimum = min(X_train)
        else:
            if isinstance(self.train_size, float):
                train_size = int(len(X) * self.train_size)
            else:
                train_size = self.train_size
            splitter = TimeSeriesSplit(n_splits=2, max_train_size=train_size)
            for tr_i, te_i in splitter.split(X):
                X_train, X_test = X.iloc[tr_i], X.iloc[te_i]
                y_train, y_test = y.iloc[tr_i], y.iloc[te_i]
            maximum = max(X_train)
            minimum = min(X_train)
        return (X_train, y_train), (X_test, y_test), (minimum, maximum)

    def load(self, filename: str = None, path: str = "."):
        """
        Load a data file.
        data_file: the data file to load
        """
        data_file = os.path.join(path, filename)
        from .data import Data

        logger.debug("Loading data")
        # load the data
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        assert isinstance(
            self, Data
        ), "Data is not an instance of Data. It is type: {}".format(type(self))
        logger.info("Loaded model")
        return data

    def save(self, filename: str, path: str = None) -> None:
        """
        Save a data file.
        filename: the filename to save the data file as
        """
        if path is None or False:
            path = os.getcwd()
        if not os.path.isdir(path):
            os.mkdir(path)
        data_file = os.path.join(path, filename)
        logger.debug("Saving data")
        # save the data
        if not hasattr(self, "X_train"):
            self()
        with open(data_file, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved model")
