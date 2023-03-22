import collections
import logging
import pickle
from typing import Tuple
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.datasets import (
    load_boston,
    load_diabetes,
    load_iris,
    load_wine,
    make_blobs,
    make_classification,
    make_moons,
    make_sparse_coded_signal,
    make_regression,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from .utils import factory
from .hashable import BaseHashable


logger = logging.getLogger(__name__)


generate = {
    "blobs": make_blobs,
    "moons": make_moons,
    "classification": make_classification,
    "signal": make_sparse_coded_signal,
    "regression": make_regression,
}
real = {
    "boston": load_boston,
    "iris": load_iris,
    "diabetes": load_diabetes,
    "wine": load_wine,
}


class Data(
    collections.namedtuple(
        typename="Data",
        field_names="name, sample, generate, real, add_noise, transform, sklearn_pipeline, target",
        defaults=(
            {},
            {},
            {},
            {},
            {},
            [],
            None,
        ),
    ),
    BaseHashable,
):
    def __new__(cls, loader, node):
        """Generates a new Data object from a YAML node"""
        return super().__new__(cls, **loader.construct_mapping(node))

    def load(self, filename):
        """
        Load data from sklearn.datasets, sklearn.datasets.make_*, csv, json, npz, or pickle as specified in params.yaml
        :return: Namespace object with X_train, X_test, y_train, y_test
        """

        params = deepcopy(self._asdict())
        if Path(filename).exists():
            ns = self.read(filename)
        elif  len(params["generate"]) > 0:
            ns = self.sklearn_load()
        else:
            ns = self.read(self.name)
        ns = self.modify(ns)            
        return ns

    def modify(self, data: Namespace) -> Namespace:
        """
        Modify the data according to the parameters in the configuration file.
        Args:
            data (Namespace): Namespace containing the data. Either "X" and "y" or "X_train", "X_test", "y_train", "y_test". If "X" and "y" are present, they are split into train and test sets.
            self.sklearn_pipeline (OrderedDict): OrderedDict of sklearn transformers to apply to the data
            self.generate (dict): Dictionary of parameters to pass to sklearn.datasets.make_*
            self.sample (dict): Dictionary of parameters to pass to sklearn.model_selection.train_test_split
            self.add_noise (dict): Dictionaries of parameters to pass to sklearn.datasets.make_sparse_coded_signal
            self.transform (dict): Dictionary of transformers to apply to the data
        Returns:
            Namespace: Namespace containing the data. Either "X" and "y" or "X_train", "X_test", "y_train", "y_test"
        """
        params = deepcopy(self._asdict())
        if "sample" in params and "X_test" not in data:
            ns = self.sampler(data)
        else:
            assert "X_test" in data, "X_test is not in data"
            assert "y_test" in data, "y_test is not in data"
            assert "X_train" in data, "X_train is not in data"
            assert "y_train" in data, "y_train is not in data"
            ns = data
        ns = self.run_sklearn_pipeline(ns)
        ns = self.add_noise_to_data(ns)
        return ns

    def sklearn_load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load  data from sklearn.datasets.make_* or sklearn.datasets.load_*, according to the name specified in params.yaml and the entry in the generate or real dictionary.
        :param self.name: Name of the dataset to load
        :return: Tuple of X, y
        """
        name = self.name
        if isinstance(name, list):
            name = name[0]
        elif not isinstance(name, str) and isinstance(eval(name), list):
            name = eval(name)[0]
        # If the data is among the sklearn "real" datasets
        if name in real:
            big_X, big_y = real[name](return_X_y=True, **self.real)
        # If the data is among the sklearn "generate" datasets
        elif name in generate:
            assert self.generate is not None, ValueError(
                "generate datasets requires a dictionary of parameters named 'generate' in params.yaml",
            )
            for k, v in self.generate.items():
                if str(v).capitalize in ["None", "False", "Null", ""]:
                    self.generate.pop(k)
            big_X, big_y = generate[name](**self.generate)
        else:
            raise ValueError(f"Unknown dataset: {name}")
        ns = Namespace(X=big_X, y=big_y)
        return ns

    def read(self, filename) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read data from csv, json, npz, or pickle as specified in params.yaml
        :param self.name: Path to the data file
        :return Tuple of X, y
        """
        name = Path(filename)
        filetype = name.suffix.replace(".","")
        params = deepcopy(self._asdict())
        # If the data is a csv file
        if filetype == "csv":
            assert "target" != None, "target column must be specified"
            df = pd.read_csv(name)
            big_X = df.drop(params["target"], axis=1)
            big_y = df[params["target"]]
        # If the data is a json
        elif filetype == "json":
            assert "target" in params, "target column must be specified"
            data = pd.read_json(name)
            if "X" in data:
                big_X = data.X
                big_y = data.y
            elif "X_train" in data:
                X_train = data.X_train
                y_train = data.y_train
                X_test = data.X_test
                y_test = data.y_test
            else:
                raise ValueError(
                    "JSON file must contain X and y attributes or X_train, y_train, X_test, and y_test attributes.",
                )
        # If the data is a numpy npz file
        elif filetype == "npz":
            data = np.load(name, allow_pickle=True)
            if "X" in data:
                big_X = data.X
                big_y = data.y
            elif "X_train" in data:
                X_train = data.X_train
                y_train = data.y_train
                X_test = data.X_test
                y_test = data.y_test
            else:
                raise ValueError(
                    "Numpy npz file must contain X and y attributes or X_train, y_train, X_test, and y_test attributes.",
                )
        elif filetype == "pickle" or filetype == "pkl":
            with open(name, "rb") as f:
                data = pickle.load(f)
            if "X" in data:
                big_X = data.X
                big_y = data.y
            elif "X_train" in data:
                X_train = data.X_train
                X_test = data.X_test
                y_train = data.y_train
                y_test = data.y_test
            else:
                raise ValueError(
                    "Pickle file must contain X and y attributes or X_train, y_train, X_test, and y_test attributes.",
                )
        else:
            raise ValueError(f"Unknown datatype: {filetype}")
        if "big_X" in locals():
            assert "big_y" in locals(), f"big_y must be specified in {name}"
            ns = Namespace(X=big_X, y=big_y)
        elif "X_train" in locals():
            assert "y_train" in locals(), f"y_train must be specified in {name}"
            assert "X_test" in locals(), f"X_test must be specified in {name}"
            assert "y_test" in locals(), f"y_test must be specified in {name}"
            ns = Namespace(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
        else:
            raise ValueError(
                f"""Data not found in correct format. Check {name}. Should be a pickle file or a csv, json, npz, or txt file. It is type {Path(name).suffix}.""",
            )
        return ns

    def add_noise_to_data(
        self,
        data: Namespace,
    ) -> Namespace:
        """
        Adds noise to the data according to the parameters specified in params.yaml
        :param add_noise.train_noise (bool): Whether to add noise to the training data
        :param add_noise.test_noise (bool): Whether to add noise to the test data
        :param add_noise.y_train_noise (float, int): Amount of noise to add to the training labels
        :param add_noise.y_test_noise (float, int): Amount of noise to add to the test labels
        """
        ###########################################################
        # Adding Noise
        ###########################################################
        data = vars(data)
        add_noise = deepcopy(dict(self.add_noise))
        train_noise = add_noise.pop("train_noise", 0)
        test_noise = add_noise.pop("test_noise", 0)
        y_train_noise = add_noise.pop("y_train_noise", 0)
        y_test_noise = add_noise.pop("y_test_noise", 0)
        X_train = data.pop("X_train", None)
        y_train = data.pop("y_train", None)
        X_test = data.pop("X_test", None)
        y_test = data.pop("y_test", None)
        # additive noise
        if train_noise != 0:
            X_train += np.random.normal(0, train_noise, X_train.shape)
        if test_noise != 0:
            X_test += np.random.normal(0, test_noise, X_test.shape)
        if y_train_noise != 0:
            if isinstance(y_train_noise, int):
                y_train += np.random.randint(0, y_train_noise, y_train.shape)
            elif isinstance(y_train_noise, float):
                y_train += np.random.normal(0, y_train_noise, y_train.shape)
            else:
                raise TypeError(
                    f"y_train_noise must be int or float, not {type(y_train_noise)}",
                )
        if y_test_noise != 0:
            if isinstance(y_test_noise, int):
                y_test += np.random.randint(0, y_test_noise, y_test.shape)
            elif isinstance(y_test_noise, float):
                y_test += np.random.normal(0, y_test_noise, y_test.shape)
            else:
                raise TypeError(
                    f"y_test_noise must be int or float, not {type(y_test_noise)}",
                )
        ns = Namespace(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        return ns

    def sklearn_transform(
        self,
        data: Namespace,
        transform: dict,
    ) -> Namespace:
        """
        Transforms the data according to the parameters specified in params.yaml
        :param data (Namespace): Namespace containing X_train, X_test, y_train, y_test
        :param name (str): Name of the transformation
        :return: Namespace containing X_train, X_test, y_train, y_test
        """
        new_data = deepcopy(data)
        X_train_bool = transform.pop("X_train", False)
        X_test_bool = transform.pop("X_test", False)
        y_train_bool = transform.pop("y_train", False)
        y_test_bool = transform.pop("y_test", False)
        object_name = transform.pop("name")
        transformer = factory(object_name, **transform)
        X_train = deepcopy(data.X_train)
        y_train = deepcopy(data.y_train)
        X_test = deepcopy(data.X_test) if hasattr(data, "X_test") else None
        y_test = deepcopy(data.y_test) if hasattr(data, "y_test") else None
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        if X_train_bool or X_test_bool is True:
            transformer.fit(X_train, y_train)
            if X_train_bool is True:
                new_X_train = transformer.transform(X_train, copy=True)
                new_data.X_train = new_X_train
            if X_test_bool is True:
                new_X_test = transformer.transform(X_test, copy=True)
                new_data.X_test = new_X_test
        if y_train_bool or y_test_bool is True:
            transformer.fit(y_train)
            if y_train_bool is True:
                new_y_train = transformer.transform(y_train, copy=True)
                new_data.y_train = new_y_train
            if y_test_bool is True:
                new_y_test = transformer.transform(y_test, copy=True)
                new_data.y_test = new_y_test
        del data
        return new_data

    def run_sklearn_pipeline(self, data) -> Namespace:
        """
        Runs the sklearn pipeline specified in params.yaml
        :param data (Namespace): Data to be transformed
        :param self.sklearn_pipeline: list of sklearn transformers
        :param self.transform: dictionary of transformers to apply to the data
        """
        pipeline = self.sklearn_pipeline
        for layer in pipeline:
            logger.info(f"Running layer {layer} of the sklearn pipeline")
            new_data = deepcopy(data)
            transform = deepcopy(self.sklearn_pipeline[layer])
            data = self.sklearn_transform(data=new_data, transform=transform)
        return data

    def sampler(
        self,
        data: Namespace,
    ) -> Namespace:
        """
        Samples the data using train_test_split
        :param data.X (np.ndarray): data
        :param data.y (np.ndarray): labels
        :param self.sample: Dictionary of parameters for train_test_split
        :return Tuple of X_train, X_test, y_train, y_test
        """
        samples = deepcopy(dict(self.sample))
        data = vars(data)
        y = data.pop("y", None)
        X = data.pop("X", None)
        ########################################################
        # Sample params
        ########################################################
        time_series = samples.pop("time_series", False)
        if "stratify" in samples and samples["stratify"] is True:
            samples.pop("stratify")
            stratify = y
        else:
            samples.pop("stratify", None)
            stratify = None
        gap = samples.pop("gap", 0)
        time_series = samples.pop("time_series", False)
        random_state = samples.pop("random_state", 0)
        train_size = samples.pop("train_size", .8)
        test_size = samples.pop("test_size", 1 - train_size)
        if isinstance(train_size, float):
            train_size = int(round(len(X) * train_size))
        if isinstance(test_size, float):
            test_size = int(round(len(X) * test_size))
        ###########################################################
        # Sampling
        ###########################################################
        # regular test/train split
        if time_series is False:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=stratify, **samples, train_size=train_size
            )
            X_test = X_test[test_size:]
            y_test = y_test[test_size:]
        # timeseries split
        elif time_series is True:
            assert (
                "test_size" or "train_size" in samples
            ), "if time series, test_size must be specified"
            max_train_size = (
                samples.pop("train_size")
                if "train_size" in samples
                else int(round(len(X) * 0.8))
            )
            assert isinstance(gap, int), "gap must be an integer"
            test_size = (
                samples.pop("test_size")
                if "test_size" in samples
                #
                else int(round(len(X) / 2 + gap))
            )
            splitter = TimeSeriesSplit(
                n_splits=2,
                max_train_size=max_train_size,
                test_size=test_size,
                gap=gap,
            )
            initial = 0
            assert initial < len(X), ValueError(
                "random_state is used to select the index of the of a subset of time series data and must be less than the length of said data + test_size",
            )
            for tr_idx, te_idx in splitter.split(X[random_state:-1]):
                X_train, X_test = X[tr_idx], X[te_idx]
                y_train, y_test = y[tr_idx], y[te_idx]
        else:
            raise ValueError(f"time_series must be True or False, not {time_series}")
        ns = Namespace(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        return ns

    def save(self, data: Namespace, filename) -> Path:
        """
        Saves the data to a pickle file
        :param data (Namespace): Namespace containing the data
        :param filename (str): Name of the file to save the data to
        :return Path to the saved data
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        return Path(filename).resolve()


config = """
    sample:
        shuffle : True
        random_state : 42
        train_size : 800
        stratify : True
    add_noise:
        train_noise : 1
    name: classification
    generate:
        n_samples: 1000
        n_features: 2
        n_informative: 2
        n_redundant : 0
        n_classes: 3
        n_clusters_per_class: 1
    sklearn_pipeline:
        scaling :
            name : sklearn.preprocessing.StandardScaler
            with_mean : true
            with_std : true
"""
