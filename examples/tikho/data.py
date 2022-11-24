import collections
import pickle
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import (
    load_boston,
    load_diabetes,
    load_iris,
    load_wine,
    make_blobs,
    make_classification,
    make_moons,
    make_sparse_coded_signal,
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from yellowbrick.features import (
    RadViz,
    Rank1D,
    Rank2D,
    PCA,
    Manifold,
    ParallelCoordinates,
)
from yellowbrick.target import ClassBalance, FeatureCorrelation

generated = {
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
        field_names="params, generated",
        defaults=({},),
    ),
):
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    def load(self):
        """
        Load data from sklearn.datasets, sklearn.datasets.make_*, csv, json, npz, or pickle as specified in params.yaml
        :return: Namespace object with X_train, X_test, y_train, y_test
        """
        # If the data is among the sklearn "real" datasets
        if self.params["name"] in real:
            big_X, big_y = real[self.params["name"]](return_X_y=True)
        # If the data is among the sklearn "generated" datasets
        elif self.params["name"] in generated:
            assert self.generated is not None, ValueError(
                "generated datasets require the generated parameter",
            )
            kwargs = self.generated
            big_X, big_y = generated[self.params["name"]](**kwargs)
        # If the data is a csv file
        elif (
            isinstance(self.params["name"], Path)
            and self.params["name"].exists()
            and str(self.params["name"]).endswith(".csv")
        ):
            assert "target" in self.params, "target column must be specified"
            df = pd.read_csv(self.params["name"])
            big_X = df.drop(self.params["target"], axis=1)
            big_y = df[self.params["target"]]
        # If the data is a json
        elif (
            isinstance(self.params["name"], Path)
            and self.params["name"].exists()
            and str(self.params["name"]).endswith(".json")
        ):
            assert "target" in self.params, "target column must be specified"
            data = pd.read_json(self.params["name"])
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
        # If the data is a numpy npz file
        elif (
            isinstance(self.params["name"], Path)
            and self.params["name"].exists()
            and str(self.params["name"]).endswith(".npz")
        ):
            data = np.load(self.params["name"])
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
        # If the data is a pickle file
        elif (
            isinstance(self.params["name"], Path)
            and self.params["name"].exists()
            and str(self.params["name"]).endswith(".pkl")
        ):
            with open(self.params["name"], "rb") as f:
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
            raise ValueError(f"Unknown dataset: {self.params['name']}")

        ########################################################
        # Optional params
        ########################################################
        # train noise
        if "train_noise" in self.params:
            train_noise = self.params.pop("train_noise")
        else:
            train_noise = 0
        # Add test noise if specified
        if "test_noise" in self.params:
            test_noise = self.params.pop("test_noise")
        else:
            test_noise = 0
        # stratify the sample, if specified
        if "stratify" in self.params and self.params["stratify"] is True:
            self.params["stratify"] = big_y
        # checking for time series
        if "time_series" in self.params and self.params["time_series"] is True:
            time_series = self.params.pop("time_series")
        else:
            time_series = False
        ###########################################################
        # Sampling
        ###########################################################
        # regular test/train split
        if "test_size" != 0 or "train_size" != 0:
            if time_series is False:
                X_train, X_test, y_train, y_test = train_test_split(
                    big_X, big_y, **self.params
                )
            # timeseries split
            elif time_series is True:
                assert (
                    "test_size" or "train_size" in self.params
                ), "if time series, test_size must be specified"
                max_train_size = (
                    self.params.pop("train_size")
                    if "train_size" in self.params
                    else int(round(len(big_X) * 0.8))
                )
                gap = 0 if "gap" not in self.params else self.params.pop("gap")
                test_size = (
                    self.params.pop("test_size")
                    if "test_size" in self.params
                    else int(round(len(big_X) / (2 + 1 + gap)))
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
        ###########################################################
        # Adding Noise
        ###########################################################
        # additive noise
        if train_noise != 0:
            X_train += np.random.normal(0, train_noise, X_train.shape)
        if test_noise != 0:
            X_test += np.random.normal(0, test_noise, X_test.shape)
        # creating namespace
        ns = Namespace(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        return ns

    def save(sself, data: Namespace, filename: str) -> Path:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        return Path(filename).resolve()

    def visualize(
        self,
        data: Namespace,
        files: dict,
        plots: dict,
        classes: list = None,
        features: list = None,
    ):
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
                visualizer = ParallelCoordinates(
                    classes=classes,
                    features=list(range(features)),
                )
                visualizer.fit(X_train, y_train)
                visualizer.show(Path(files["path"], plots["parallel"]))
                paths.append(Path(files["path"], plots["parallel"]))
                plots.pop("parallel")
                plt.gcf().clear()
        return paths


if "__main__" == __name__:
    data_document = """
        name: blobs
        params:
            shuffle : True
            random_state : 42
            train_size : 800
            stratify : True
            train_noise : 1
            time_series : True
        generated:
            n_samples: 1000
            n_features: 2
            centers: 2
    """
    yaml.add_constructor("!Data:", Data)
    data_document_tag = """!Data:""" + data_document
    # Test that data yaml loads correctly
    data = yaml.load(data_document_tag, Loader=yaml.Loader)
    data = data.load()
    assert "X_train" in data
    assert "y_train" in data
    assert "y_test" in data
    assert "X_test" in data
