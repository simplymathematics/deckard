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
from yellowbrick.features import (PCA, Manifold, ParallelCoordinates, RadViz,
                                  Rank1D, Rank2D)
from yellowbrick.target import ClassBalance, FeatureCorrelation

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
        field_names="name, sample, files, generate, real, transform, sklearn_pipeline, art_pipeline",
        defaults=({},{},{},[],[],),
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
            big_X, big_y = self.sklearn_load(name)
        elif isinstance(name, Path) and name.exists() and not str(name).endswith(".pkl") and not str(name).endswith(".pickle"):
            big_X, big_y = self.read(name)
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
        X_train, X_test, y_train, y_test = self.sample(big_X, big_y, **samples)
        X_train, X_test, y_train, y_test = self.add_noise(X_train, X_test, y_train, y_test, **params)
        ns = Namespace(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
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
        return 
    
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
            raise ValueError(f"Unknown datatype: {name.split(".")[-1]}")
        
        return big_X, big_y
    
    def add_noise(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
        """
        Adds noise to the data according to the parameters specified in params.yaml
        :param X_train (np.ndarray): Training data
        :param X_test (np.ndarray): Testing data
        :param y_train (np.ndarray): Training labels
        :param y_test (np.ndarray): Testing labels
        :param self.params.train_noise : Noise to add to the training data
        :param self.params.test_noise : Noise to add to the testing data
        :param self.params.y_train_noise : Noise to add to the training labels 
        :param self.params.y_test_noise : Noise to add to the testing labels
        """
        ###########################################################
        # Adding Noise
        ###########################################################
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

    
    
    def sample(self, X:np.ndarray, y:np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        samples = kwargs
        ########################################################
        # Sample params
        ########################################################
        train_noise = samples.pop("train_noise", 0)
        test_noise = samples.pop("test_noise", 0)
        test_noise = 0
        gap = samples.pop("gap", 0)
        time_series = samples.pop("time_series", False)
        if "stratify" in params and params["stratify"] is True:
            samples.pop("stratify")
            stratify = big_y
        else:
            stratify =False
        ###########################################################
        # Sampling
        ###########################################################
        # regular test/train split
        if "test_size" in samples or "train_size" in samples:
            if time_series is False:
                X_train, X_test, y_train, y_test = train_test_split(
                    big_X, big_y, stratify, **samples
                )
            # timeseries split
            elif time_series is True:
                assert (
                    "test_size" or "train_size" in samples
                ), "if time series, test_size must be specified"
                max_train_size = (
                    samples.pop("train_size")
                    if "train_size" in samples
                    else int(round(len(big_X) * 0.8))
                )
                assert isinstance(gap, int), "gap must be an integer"
                test_size = (
                    samples.pop("test_size")
                    if "test_size" in samples
                    # 
                    else int(round(len(big_X) / 2 + gap))
                )
                splitter = TimeSeriesSplit(
                    n_splits=2,
                    max_train_size=max_train_size,
                    test_size=test_size,
                    gap=gap,
                )
                initial = 0
                assert initial < len(big_X), ValueError(
                    "random_state is used to select the index of the of a subset of time series data and must be less than the length of said data + test_size",
                )
                for tr_idx, te_idx in splitter.split(big_X):
                    X_train, X_test = big_X[tr_idx], big_X[te_idx]
                    y_train, y_test = big_y[tr_idx], big_y[te_idx]
        return X_train, X_test, y_train, y_test

    def save(self, data: Namespace) -> Path:
        filename = Path(self.files['data_path'],  my_hash(self._asdict()) + "." + self.files['data_filetype'])
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        return Path(filename).resolve()

    def visualize(self, data: Namespace, files: dict, plots: dict, classes:list = None, features:list = None) -> List[Path]:
        y_train = data.y_train
        X_train = data.X_train
        classes = set(y_train) if classes is None else classes
        features = X_train.shape[1] if features is None else features
        paths = []
        if len(plots.keys()) > 0:
            assert (
                "path" in files
            ), "Path to save plots is not specified in the files entry in params.yaml"
            Path(files["path"]).mkdir(parents=True, exist_ok=True)
            if "radviz" in plots:
                visualizer = RadViz(classes=classes)
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["radviz"]))
                paths.append(Path(files["path"], plots["radviz"]))
                plots.pop("radviz")
                plt.gcf().clear()
            if "rank1d" in plots:
                visualizer = Rank1D(algorithm="shapiro")
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["rank1d"]))
                paths.append(Path(files["path"], plots["rank1d"]))
                plots.pop("rank1d")
                plt.gcf().clear()
            if "rank2d" in plots:
                visualizer = Rank2D(algorithm="pearson")
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["rank2d"]))
                paths.append(Path(files["path"], plots["rank2d"]))
                plots.pop("rank2d")
                plt.gcf().clear()
            if "balance" in plots:
                visualizer = ClassBalance(labels=classes)
                visualizer.fit(y_train)
                visualizer.show(Path(files["path"], plots["balance"]))
                paths.append(Path(files["path"], plots["balance"]))
                plots.pop("balance")
                plt.gcf().clear()
            if "correlation" in plots:
                visualizer = FeatureCorrelation(labels=list(range(features)))
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["correlation"]))
                paths.append(Path(files["path"], plots["correlation"]))
                plots.pop("correlation")
                plt.gcf().clear()
            if "pca" in plots:
                visualizer = PCA()
                visualizer.fit_transform(X_train, y_train)
                visualizer.show(Path(files["path"], plots["pca"]))
                paths.append(Path(files["path"], plots["pca"]))
                plots.pop("pca")
                plt.gcf().clear()
            if "manifold" in plots:
                visualizer = Manifold(manifold="tsne")
                visualizer.fit_transform(X_train, y_train)
                visualizer.show(Path(files["path"], plots["manifold"]))
                paths.append(Path(files["path"], plots["manifold"]))
            if "parallel" in plots:
                visualizer = ParallelCoordinates(classes = classes, features = list(range(features)))
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["parallel"]))
                paths.append(Path(files["path"], plots["parallel"]))
                plots.pop("parallel")
                plt.gcf().clear()
        return paths


if "__main__" == __name__:
    data_document = """
        sample:
            shuffle : True
            random_state : 42
            train_size : 800
            stratify : True
            train_noise : 1
            time_series : True
        name: iris
        files:
            data_path : data
            data_filetype : pickle
        generate:
            n_samples: 1000
            n_features: 2
            centers: 2
        
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
