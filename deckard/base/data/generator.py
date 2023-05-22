import logging
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_blobs,
    make_moons,
    make_circles,
    make_biclusters,
)
from typing import Literal
from dataclasses import dataclass, field
from ..utils import my_hash
import numpy as np
from art.utils import load_mnist, load_cifar10

__all__ = [
    "SklearnDataGenerator",
    "TorchDataGenerator",
    "KerasDataGenerator",
    "DataGenerator",
]
logger = logging.getLogger(__name__)


@dataclass
class SklearnDataGenerator:
    name: Literal["classification", "regression"] = "classification"
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
        elif self.name in "biclusters":
            X, y = make_biclusters(**self.kwargs)
        else:
            raise ValueError(f"Unknown dataset name {self.name}")
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


@dataclass
class TorchDataGenerator:
    name: Literal["torch_mnist", "torch_cifar10"] = "torch_mnist"
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
        else:
            raise ValueError(f"Unknown dataset name {self.name}")
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


@dataclass
class KerasDataGenerator:
    name: Literal["mnist", "cifar10"] = "mnist"
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
        else:
            raise ValueError(f"Unknown dataset name {self.name}")
        return [X, y]

    def __hash__(self):
        return int(my_hash(self), 16)


@dataclass
class DataGenerator:
    name: str = "classification"
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = {k: v for k, v in kwargs.items() if v is not None}

    def __call__(self):
        if self.name in ["classification", "regression"]:
            return SklearnDataGenerator(self.name, **self.kwargs)()
        elif self.name in ["torch_mnist", "torch_cifar10"]:
            return TorchDataGenerator(self.name, **self.kwargs)()
        elif self.name in ["keras_mnist", "keras_cifar10", "mnist", "cifar10"]:
            return KerasDataGenerator(self.name, **self.kwargs)()
        else:
            raise ValueError(f"Invalid name {self.name}. Please choose from ")

    def __hash__(self):
        return int(my_hash(self), 16)
