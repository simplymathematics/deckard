import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union
import numpy as np
from art.estimators.classification import KerasClassifier
from art.estimators.regression import KerasRegressor

logger = logging.getLogger(__name__)


classifier_dict = {
    "keras": KerasClassifier,
}

regressor_dict = {
    "keras-regressor": KerasRegressor,
}

keras_dict = {**classifier_dict, **regressor_dict}
keras_models = list(keras_dict.keys())
__all__ = ["KerasInitializer", "KerasLoss", "KerasOptimizer"]


@dataclass
class KerasLoss:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self):
        import tensorflow as tf

        if len(self.kwargs) > 0:
            config = {"class_name": self.name, "config": self.kwargs}
        else:
            config = self.name
        obj = tf.keras.losses.get(config)
        return obj


@dataclass
class KerasInitializer:
    model: object
    data: list
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, data, model, library, **kwargs):
        self.data = data
        self.model = model
        self.library = library
        kwargs.pop("library", None)
        kwargs.pop("data", None)
        kwargs.pop("model", None)
        i = 0
        while "kwargs" in kwargs and i < 10:
            kwargs.update(**kwargs.pop("kwargs", {}))
            i += 1
        self.kwargs = kwargs

    def __call__(self):
        kwargs = deepcopy(self.kwargs)
        model = self.model
        data = self.data
        library = self.library
        kwargs
        if hasattr(model, "metrics"):
            metrics = model.metrics
        elif hasattr(model, "model") and hasattr(model.model, "metrics"):
            metrics = model.model.metrics
        else:
            raise ValueError("model must have metrics attribute")
        if isinstance(metrics, tuple):
            metrics = list(*metrics)
        if not isinstance(metrics, list):
            metrics = [metrics]
        if hasattr(model, "loss"):
            loss = model.loss
        elif hasattr(model, "model") and hasattr(model.model, "loss"):
            loss = model.model.loss
        else:
            raise ValueError("model must have loss attribute")
        if hasattr(model, "optimizer"):
            optimizer = model.optimizer
        elif hasattr(model, "model") and hasattr(model.model, "optimizer"):
            optimizer = model.model.optimizer
        else:
            raise ValueError("model must have optimizer attribute")
        if hasattr(model, "model"):
            model = model.model
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        if "clip_values" not in kwargs:
            X_train, _, _, _ = data
            kwargs.update({"clip_values": (np.amin(X_train), np.amax(X_train))})
        if "preprocessing" not in kwargs:
            if data[0].shape[-1] > 1:
                mean = np.mean(data[0], axis=0)
                std = np.std(data[0], axis=0)
                pre_tup = (mean, std)
            else:
                pre_tup = (np.mean(data[0]), np.std(data[0]))
            kwargs.update({"preprocessing": pre_tup})
        if library in keras_dict:
            if "library" in kwargs:
                kwargs.pop("library")
            if str(type(model)).startswith("<class 'art."):
                model = keras_dict[library](model.model, **kwargs)
            else:
                model = keras_dict[library](model, **kwargs)
        else:
            raise ValueError(f"library must be one of {keras_models}. Got {library}")
        return model


@dataclass
class KerasOptimizer:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self):
        import tensorflow as tf
        if len(self.kwargs) > 0:
            config = {"class_name": self.name, "config": self.kwargs}
        else:
            config = self.name
        obj = tf.keras.optimizers.get(config)
        return obj
