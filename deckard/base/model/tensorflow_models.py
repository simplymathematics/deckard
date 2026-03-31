import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union
from art.estimators.classification import (
    TensorFlowV2Classifier,
)
from omegaconf import DictConfig, OmegaConf
import numpy as np
from ..utils import factory

logger = logging.getLogger(__name__)


tensorflow_dict = {
    "tensorflow": TensorFlowV2Classifier,
    "tensorflowv2": TensorFlowV2Classifier,
    "tf2": TensorFlowV2Classifier,
    "tfv2": TensorFlowV2Classifier,
}


tensorflow_models = list(tensorflow_dict.keys())

__all__ = ["TensorflowV2Initializer", "TensorflowV2Loss", "TensorflowV2Optimizer"]


@dataclass
class TensorflowV2Loss:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self):
        import tensorflow as tf

        tf.config.run_functions_eagerly(True)
        logger.debug(f"Initializing model {self.name} with kwargs {self.kwargs}")
        if len(self.kwargs) > 0:
            config = {"class_name": self.name, "config": self.kwargs}
        else:
            config = self.name
        obj = tf.keras.losses.get(config)
        return obj


@dataclass
class TensorflowV2Initializer:
    data: list
    model: str
    library: str = "tensorflow"
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, data, model, library="tensorflow", **kwargs):
        self.data = data
        self.model = model
        library = library
        while "kwargs" in kwargs:
            kwargs.update(**kwargs.pop("kwargs", {}))
        self.kwargs = kwargs

    def __call__(self):
        import tensorflow as tf

        tf.config.run_functions_eagerly(True)
        kwargs = deepcopy(self.kwargs)
        data = self.data
        model = self.model
        library = self.library
        loss = kwargs.pop("loss", "categorical_crossentropy")
        optimizer = kwargs.pop("optimizer", "adam")
        if isinstance(optimizer, DictConfig):
            optimizer = OmegaConf.to_container(optimizer, resolve=True)
        if isinstance(optimizer, dict):
            name = optimizer.pop("name")
            params = {**optimizer}
        elif isinstance(optimizer, str):
            name = optimizer
            params = {}
        else:
            raise ValueError(
                f"optimizer must be a dict or str or DictConfig. Got {type(optimizer)} for optimizer: {optimizer}",
            )
        optimizer = tf.keras.optimizers.get(name, **params)
        if isinstance(loss, DictConfig):
            loss = OmegaConf.to_container(loss, resolve=True)
        if isinstance(loss, dict):
            name = loss.pop("name")
            params = {**loss}
        elif isinstance(loss, str):
            name = loss
            params = {}
        else:
            raise ValueError(
                f"loss must be a dict or str or DictConfig. Got {type(loss)} for loss: {loss}",
            )
        loss = tf.keras.losses.get(name, **params)
        if "preprocessing" not in kwargs:
            if data[0].shape[-1] > 1:
                mean = np.mean(data[0], axis=0)
                std = np.std(data[0], axis=0)
                pre_tup = (mean, std)
            else:
                pre_tup = (np.mean(data[0]), np.std(data[0]))
            kwargs.update({"preprocessing": pre_tup})
        if "clip_values" not in kwargs:
            clip_values = (np.min(data[0]), np.max(data[0]))
            kwargs.update({"clip_values": clip_values})
        if "nb_classes" not in kwargs:
            if len(np.squeeze(data[2]).shape) > 1:
                nb_classes = len(np.unique(np.argmax(np.squeeze(data[2]), axis=1)))
            else:
                nb_classes = len(np.unique(np.squeeze(data[2])))
            kwargs.update({"nb_classes": nb_classes})
        if "input_shape" not in kwargs:
            input_shape = data[0][0].shape
            kwargs.update({"input_shape": input_shape})
        if "train_step" not in kwargs:
            assert hasattr(model, "train_step"), "Model must have train_step attribute"
            train_step = model.train_step
        else:
            train_step = kwargs.pop("train_step")
            train_step = factory(name=train_step.pop("name"), **train_step)
        if library in tensorflow_dict and not isinstance(
            model,
            tuple(tensorflow_dict.values()),
        ):
            est = tensorflow_dict[library]
            model = est(model, **kwargs, train_step=train_step)
        elif isinstance(model, tuple(tensorflow_dict.values())):
            est = model.model
            model = est(model, **kwargs, train_step=train_step)
        else:
            raise ValueError(
                f"library must be one of {tensorflow_models}. Got {library}",
            )
        return model


@dataclass
class TensorflowV2Optimizer:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self):
        import tensorflow as tf

        tf.config.run_functions_eagerly(True)
        logger.debug(f"Initializing model {self.name} with kwargs {self.kwargs}")
        if "kwargs" in self.kwargs:
            kwargs = self.kwargs.pop("kwargs", {})
            params = self.kwargs
            params.pop("name", None)
            params.update(**kwargs)
        else:
            params = self.kwargs
        obj = tf.keras.optimizers.get(self.name, **params)
        return obj


@dataclass
class TensorflowV1Loss:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self):
        import tensorflow as tf

        tf.config.run_functions_eagerly(True)
        logger.debug(f"Initializing model {self.name} with kwargs {self.kwargs}")
        if "kwargs" in self.kwargs:
            kwargs = self.kwargs.pop("kwargs", {})
            params = self.kwargs
            params.pop("name", None)
            params.update(**kwargs)
        else:
            params = self.kwargs
        obj = tf.keras.losses.get(self.name, **params)
        return obj


@dataclass
class TensorflowV1Optimizer:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self):
        import tensorflow as tf

        logger.debug(f"Initializing model {self.name} with kwargs {self.kwargs}")
        if "kwargs" in self.kwargs:
            kwargs = self.kwargs.pop("kwargs", {})
            params = self.kwargs
            name = params.pop("name", None)
            params.update(**kwargs)
        else:
            name = self.name
            params = self.kwargs
        obj = tf.keras.optimizers.get(name, **params)
        return obj
