import logging

from typing import Literal
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from validators import url
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_blobs,
    make_moons,
    make_circles,
)
from art.utils import load_mnist, load_cifar10, load_diabetes, to_categorical
from ..utils import my_hash

__all__ = [
    "SklearnDataGenerator",
    "TorchDataGenerator",
    "KerasDataGenerator",
    "DataGenerator",
]
logger = logging.getLogger(__name__)

SKLEARN_DATASETS = [
    "classification",
    "regression",
    "blobs",
    "moons",
    "circles",
]


@dataclass
class SklearnDataGenerator:
    name: Literal[
        "classification",
        "regression",
        "blobs",
        "moons",
        "circles",
        "biclusters",
    ] = "classification"
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and kwargs={kwargs}",
        )
        self.name = name
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self):
        if self.name in "classification":
            X, y = make_classification(**self.kwargs)
        elif self.name in "regression":
            X, y = make_regression(**self.kwargs)
        elif self.name in "blobs":
            X, y = make_blobs(**self.kwargs)
        elif self.name in "moons":
            X, y = make_moons(**self.kwargs)
        elif self.name in "circles":
            X, y = make_circles(**self.kwargs)
        elif isinstance(self.name, str) and Path(self.name).exists():
            suffix = Path(self.name).suffix
            if suffix == ".npz":
                with np.load(self.name) as data:
                    X = data["X"]
                    y = data["y"]
            elif suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(self.name)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif suffix == ".json":
                import json

                with open(self.name, "r") as f:
                    data = json.load(f)
                X = data["X"]
                y = data["y"]
            elif suffix in [".pkl", ".pickle"]:
                import pickle

                with open(self.name, "rb") as f:
                    data = pickle.load(f)
                X = data["X"]
                y = data["y"]
        else:  # pragma: no cover
            raise ValueError(f"Unknown dataset name {self.name}")
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


TORCH_DATASETS = ["torch_mnist", "torch_cifar10", "torch_diabetes", "torch_cifar100"]


@dataclass
class TorchDataGenerator:
    name: Literal[
        "torch_mnist",
        "torch_cifar10",
        "torch_diabetes",
        "torch_cifar100",
    ] = "torch_mnist"
    path = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, path=None, **kwargs):
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and kwargs={kwargs}",
        )
        self.name = name
        self.path = path
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self):
        if self.name == "torch_mnist":
            (X_train, y_train), (X_test, y_test), _, _ = load_mnist()
            # Append train and test data to create
            X_train = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32)
            X_test = np.transpose(X_test, (0, 3, 1, 2)).astype(np.float32)
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        elif self.name == "torch_cifar10" or self.name == "torch_cifar":
            (X_train, y_train), (X_test, y_test), _, _ = load_cifar10()
            # Append train and test data to create
            X_train = np.transpose(X_train, (0, 3, 1, 2)).astype(np.float32)
            X_test = np.transpose(X_test, (0, 3, 1, 2)).astype(np.float32)
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        elif self.name == "torch_diabetes":
            (X_train, y_train), (X_test, y_test), _, _ = load_diabetes()
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        elif self.name == "torch_cifar100":
            try:
                from torchvision.datasets import CIFAR100
                from torchvision import transforms
            except ImportError:  # pragma: no cover
                raise ImportError("Please install torchvision to use CIFAR100")
            if self.path is None:  # pragma: no cover
                raise ValueError(
                    f"path attribute must be specified for dataset: {self.name}.",
                )
            original_filename = Path(self.path, self.name, f"{self.name}.npz")
            Path(original_filename.parent).mkdir(parents=True, exist_ok=True)
            if not original_filename.exists():
                train_set = CIFAR100(
                    Path(self.path, self.name),
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                )
                test_set = CIFAR100(
                    Path(self.path, self.name),
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )

                def X_(x):
                    return np.array(x[0])

                def y_(X):
                    return np.array(X[1])

                X_train = np.array(list(map(X_, train_set)))
                y_train = np.array(list(map(y_, train_set)))
                X_test = np.array(list(map(X_, test_set)))
                y_test = np.array(list(map(y_, test_set)))
                y_train = to_categorical(y_train, 100)
                y_test = to_categorical(y_test, 100)
                X = np.concatenate((X_train, X_test))
                y = np.concatenate((y_train, y_test))
                np.savez(file=original_filename.as_posix(), X=X, y=y)
            else:
                dict_ = np.load(original_filename.as_posix())
                X = dict_["X"]
                y = dict_["y"]
        else:  # pragma: no cover
            raise ValueError(f"Unknown dataset name {self.name}")
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


KERAS_DATASETS = ["keras_mnist", "keras_cifar10", "mnist", "cifar10", "diabetes"]


@dataclass
class KerasDataGenerator:
    name: Literal["mnist", "cifar10", "diabetes"] = "mnist"
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        logger.info(
            f"Instantiating {self.__class__.__name__} with name={name} and kwargs={kwargs}",
        )
        self.name = name
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self):
        if "cifar" in self.name:
            (X_train, y_train), (X_test, y_test), _, _ = load_cifar10()
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        elif "mnist" in self.name:
            (X_train, y_train), (X_test, y_test), _, _ = load_mnist()
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        elif "diabetes" in self.name:
            (X_train, y_train), (X_test, y_test), _, _ = load_diabetes()
            X = np.concatenate((X_train, X_test))
            y = np.concatenate((y_train, y_test))
        else:  # pragma: no cover
            raise ValueError(f"Unknown dataset name {self.name}")
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


ALL_DATASETS = []
ALL_DATASETS.extend(SKLEARN_DATASETS)
ALL_DATASETS.extend(TORCH_DATASETS)
ALL_DATASETS.extend(SKLEARN_DATASETS)
ALL_DATASETS.extend(KERAS_DATASETS)


@dataclass
class DataGenerator:
    name: str = "classification"
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self):
        if self.name in SKLEARN_DATASETS:
            return SklearnDataGenerator(self.name, **self.kwargs)()
        elif self.name in TORCH_DATASETS:
            return TorchDataGenerator(self.name, **self.kwargs)()
        elif self.name in KERAS_DATASETS:
            return KerasDataGenerator(self.name, **self.kwargs)()
        elif isinstance(self.name, str) and Path(self.name).exists():
            return SklearnDataGenerator(self.name, **self.kwargs)()
        elif url(self.name):
            return SklearnDataGenerator(self.name, **self.kwargs)()
        else:  # pragma: no cover
            raise ValueError(
                f"Invalid name {self.name}. Please choose from {ALL_DATASETS}",
            )

    def __hash__(self):
        return int(my_hash(self), 16)
