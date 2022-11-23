import collections
import logging
import pickle
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from hashable import BaseHashable, my_hash
from sklearn.datasets import (load_boston, load_diabetes, load_iris, load_wine,
                              make_blobs, make_classification, make_moons,
                              make_sparse_coded_signal)
from sklearn.model_selection import TimeSeriesSplit, train_test_split


from utils import factory
from typing import Tuple, List
logger = logging.getLogger(__name__)



generate = {
    "blobs": make_blobs,
    "moons": make_moons,
    "classification": make_classification,
    "signal": make_sparse_coded_signal,
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
        field_names="name, sample, files, generate, real, add_noise, transform, sklearn_pipeline",
        defaults=({},{},{},{},[],),
    ), BaseHashable,
):
    def __new__(cls, loader, node):
        """ Generates a new Data object from a YAML node """
        return super().__new__(cls, **loader.construct_mapping(node))


    def load(self):
        """
        Load data from sklearn.datasets, sklearn.datasets.make_*, csv, json, npz, or pickle as specified in params.yaml
        :return: Namespace object with X_train, X_test, y_train, y_test
        """
        assert "name" in self._asdict(), "Name of the dataset is not specified in params.yaml"
        filename = Path(self.files['data_path'],  my_hash(self._asdict()) + "." + self.files['data_filetype'])
        params = deepcopy(self._asdict())
        if filename.exists():
            logger.info(f"Loading data from {filename}")
            name = filename
        else:
            name = params.pop("name")
        if name in real or name in generate:
            big_X, big_y = self.sklearn_load()
        elif isinstance(name, Path) and name.exists() and not str(name).endswith(".pkl") and not str(name).endswith(".pickle"):
            big_X, big_y = self.read()
        # If the data is a pickle file
        elif (
            isinstance(name, Path)
            and name.exists()
            and (str(name).endswith(".pkl") or str(name).endswith(".pickle"))
        ):
            with open(name, "rb") as f:
                data = pickle.load(f)
            if "X_train" in data:
                assert (
                    hasattr(data, "y_train")
                    and hasattr(data, "X_test")
                    and hasattr(data, "y_test")
                ), ValueError(
                    "X_train, y_train, X_test, and y_test must all be present in the pickle file",
                )
                X_train = data.X_train
                y_train = data.y_train
                X_test = data.X_test
                y_test = data.y_test
                return Namespace(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )
            else:
                assert hasattr(data, "X") and hasattr(data, "y"), ValueError(
                    "data must have X and y attributes",
                )
                big_X = data.X
                big_y = data.y
        # Otherwise, raise an error
        else:
            raise ValueError(f"Unknown dataset: {name}")
        samples = params.pop("sample", {})
        X_train, X_test, y_train, y_test = self.sampler(X = big_X, y = big_y)
        add_noise = params.pop("add_noise", {})
        X_train, X_test, y_train, y_test = self.add_noise_to_data(X_train, X_test, y_train, y_test)
        ns = Namespace(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        ns = self.run_sklearn_pipeline(ns)
        return ns
    
    def sklearn_load(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load  data from sklearn.datasets.make_* or sklearn.datasets.load_*, according to the name specified in params.yaml and the entry in the generate or real dictionary. 
        :param self.name: Name of the dataset to load
        :return: Tuple of X, y
        """
        name = self.name
        # If the data is among the sklearn "real" datasets
        if name in real:
            big_X, big_y = real[name](return_X_y=True, **self.real)
        # If the data is among the sklearn "generate" datasets
        elif name in generate:
            assert self.generate is not None, ValueError(
                "generate datasets requires a dictionary of parameters named 'generate' in params.yaml",
            )
            big_X, big_y = generate[name](**self.generate)
        return (big_X, big_y)
    
    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read data from csv, json, npz, or pickle as specified in params.yaml
        :param self.name: Path to the data file
        :return Tuple of X, y
        """
        name = self.name
        # If the data is a csv file
        if (
            str(name).endswith(".csv")
        ):
            assert "target" in params, "target column must be specified"
            df = pd.read_csv(name)
            big_X = df.drop(params["target"], axis=1)
            big_y = df[params["target"]]
        # If the data is a json
        elif (
            str(name).endswith(".json")
        ):
            assert "target" in params, "target column must be specified"
            data = pd.read_json(name)
            assert hasattr(data, "X") and hasattr(data, "y"), ValueError(
                "data must have X and y attributes",
            )
            big_X = data.X
            big_y = data.y
        # If the data is a numpy npz file
        elif (
            isinstance(name, Path)
            and name.exists()
            and str(name).endswith(".npz")
        ):
            data = np.load(name)
            assert (hasattr(data, "y"))
            big_y = data.y
            big_X = data.X
        else:
            raise ValueError(f"Unknown datatype: {name.split('.S')[-1]}")
        
        return big_X, big_y
    
    def add_noise_to_data(self, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Adds noise to the data according to the parameters specified in params.yaml
        :param X_train (np.ndarray): Training data
        :param X_test (np.ndarray): Testing data
        :param y_train (np.ndarray): Training labels
        :param y_test (np.ndarray): Testing labels
        :param self.add_noise : Noise to add to the training data
        :return: Tuple of X_train, X_test, y_train, y_test
        """
        ###########################################################
        # Adding Noise
        ###########################################################
        add_noise = self.add_noise
        train_noise = add_noise.pop("train_noise", 0)
        test_noise = add_noise.pop("test_noise", 0)
        y_train_noise = add_noise.pop("y_train_noise", 0)
        y_test_noise = add_noise.pop("y_test_noise", 0)
        gap = add_noise.pop("gap", 0)
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
                raise TypeError(f"y_train_noise must be int or float, not {type(y_train_noise)}")
        if y_test_noise !=0:
            if isinstance(y_test_noise, int):
                y_test += np.random.randint(0, y_test_noise, y_test.shape)
            elif isinstance(y_test_noise, float):
                y_test += np.random.normal(0, y_test_noise, y_test.shape)
            else:
                raise TypeError(f"y_test_noise must be int or float, not {type(y_test_noise)}")
        return X_train, X_test, y_train, y_test        

    def sklearn_transform(self, data:Namespace, transform:dict = None, name:str = "") -> Namespace:
        """
        Transofrms the data according to the parameters specified in params.yaml
        :param data (Namespace): Namespace containing X_train, X_test, y_train, y_test
        :param transform (dict): Dictionary of parameters for the transformation
        :return: Namespace containing X_train, X_test, y_train, y_test
        """
        new_data = deepcopy(data)
        if transform is None:
            transform = dict(self.transform)
        X_train_bool = transform.pop("X_train", False)
        X_test_bool = transform.pop("X_test", False)
        y_train_bool = transform.pop("y_train", False)
        y_test_bool = transform.pop("y_test", False)
        transformer = factory(name, **transform)
        X_train = deepcopy(data.X_train)
        y_train = deepcopy(data.y_train)
        X_test = deepcopy(data.X_test)
        y_test = deepcopy(data.y_test)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        if X_train_bool or X_test_bool is True:
            transformer.fit(X_train, y_train)
            if X_train_bool is True:
                new_X_train = transformer.transform(X_train, copy = True)
                new_data.X_train = new_X_train 
            if X_test_bool is True:
                new_X_test  = transformer.transform(X_test, copy = True)
                new_data.X_test  = new_X_test
        if y_train_bool or y_test_bool is True:
            transformer.fit(y_train)
            if y_train_bool is True:
                new_y_train = transformer.transform(y_train, copy = True)
                new_data.y_train = new_y_train
            if y_test_bool is True:
                new_y_test = transformer.transform(y_test, copy = True)
                new_data.y_test  = new_y_test
        del data
        return new_data
            
    def run_sklearn_pipeline(self, data):
        """
        Runs the sklearn pipeline specified in params.yaml
        :param data (Namespace): Data to be transformed
        :param self.sklearn_pipeline: list of sklearn transformers
        :param self.transform: dictionary of transformers to apply to the data
        """
        pipeline = self.sklearn_pipeline
        for layer in pipeline:
            new_data = deepcopy(data)
            transform = self.transform[layer]
            data = self.sklearn_transform(new_data, transform, name = layer)
        return data
        
    def sampler(self, X:np.ndarray, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples the data using train_test_split
        :param X (np.ndarray): data
        :param y (np.ndarray): labels
        :param self.sample: Dictionary of parameters for train_test_split
        :return Tuple of X_train, X_test, y_train, y_test
        """
        samples = dict(self.sample)
        ########################################################
        # Sample params
        ########################################################
        time_series = samples.pop("time_series", False)
        if "stratify" in samples and samples["stratify"] is True:
            samples.pop("stratify")
            stratify = y
        else:
            stratify =False
        gap = samples.pop("gap", 0)
        time_series = samples.pop("time_series", False)
        ###########################################################
        # Sampling
        ###########################################################
        # regular test/train split
        if time_series is False:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = stratify, **samples)
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
        return (X_train, X_test, y_train, y_test)

    
    
    def save(self, data: Namespace) -> Path:
        """
        Saves the data to a pickle file at self.files.data_path/hash/self.files.data_name
        :param data (Namespace): Namespace containing the data
        :param self.files.data_path : Path to save the data
        :param self.files.data_name : Name of the data
        :return Path to the saved data
        """
        filename = Path(self.files['data_path'],  my_hash(self._asdict()) + "." + self.files['data_filetype'])
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        return Path(filename).resolve()


if "__main__" == __name__:
    data_document = """
        sample:
            shuffle : True
            random_state : 42
            train_size : 800
            stratify : True
        add_noise:
            train_noise : 1
            time_series : True
        name: classification
        files:
            data_path : data
            data_filetype : pickle
        generate:
            n_samples: 1000
            n_features: 2
            n_informative: 2
            n_redundant : 0
            n_classes: 2
        sklearn_pipeline:
            - sklearn.preprocessing.StandardScaler
        transform:
            sklearn.preprocessing.StandardScaler:
                with_mean : true
                with_std : true
                X_train : true
                X_test : true
    """
    yaml.add_constructor("!Data:", Data)
    data_document_tag = """!Data:""" + data_document
    # Test that data yaml loads correctly
    data = yaml.load(data_document_tag, Loader=yaml.Loader)
    data_ = data.load()
    file1 = data.save(data_)
    data2 = yaml.load(data_document_tag, Loader=yaml.Loader)
    data2_ = data2.load()
    file2 = data2.save(data2_)
    assert "X_train" in data_
    assert file1 == file2
    assert data_.X_train.shape == data2_.X_train.shape
    assert data_.X_test.shape == data2_.X_test.shape
    assert data_.y_train.shape == data2_.y_train.shape
