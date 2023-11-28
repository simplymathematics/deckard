import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union
from random import randint
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor
from hydra.utils import instantiate


logger = logging.getLogger(__name__)

classifier_dict = {
    "pytorch": PyTorchClassifier,
    "torch": PyTorchClassifier,
}

regressor_dict = {
    "pytorch-regressor": PyTorchRegressor,
    "torch-regressor": PyTorchRegressor,
}

torch_dict = {**classifier_dict, **regressor_dict}
supported_models = list(torch_dict.keys())

__all__ = ["TorchInitializer", "TorchCriterion", "TorchOptimizer"]


@dataclass
class TorchCriterion:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = kwargs.pop("_target_", name)
        self.kwargs = kwargs

    def __call__(self):
        logger.info(f"Initializing model {self.name} with kwargs {self.kwargs}")
        if "kwargs" in self.kwargs:
            kwargs = self.kwargs.pop("kwargs", {})
            params = self.kwargs
            params.pop("name", None)
            params.update(**kwargs)
        else:
            params = self.kwargs
        name = params.pop("_target_", self.name)
        dict_ = {"_target_": name}
        dict_.update(**params)
        obj = instantiate(dict_)
        return obj


@dataclass
class TorchOptimizer:
    name: str
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, name, **kwargs):
        self.name = kwargs.pop("_target_", name)
        self.kwargs = kwargs

    def __call__(self, model):
        logger.info(f"Initializing model {self.name} with kwargs {self.kwargs}")
        if "kwargs" in self.kwargs:
            kwargs = self.kwargs.pop("kwargs", {})
            params = self.kwargs
            params.pop("name", None)
            params.update(**kwargs)
        else:
            params = self.kwargs
        name = params.pop("_target_", self.name)
        dict_ = {"_target_": name}
        dict_.update(**params)
        if hasattr(model, "parameters"):
            dict_.update({"params": model.parameters()})
        elif hasattr(model, "model") and hasattr(model.model, "parameters"):
            dict_.update({"params": model.model.parameters()})
        else:
            raise ValueError(f"Model {model} has no parameters attribute.")
        obj = instantiate(dict_)
        return obj


@dataclass
class TorchInitializer:
    data: list
    model: str
    optimizer: TorchOptimizer = field(default_factory=TorchOptimizer)
    criterion: TorchCriterion = field(default_factory=TorchCriterion)
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, data, model, library, **kwargs):
        self.data = data
        self.model = model
        self.library = library
        while "kwargs" in kwargs:
            new_kwargs = kwargs.pop("kwargs", {})
            kwargs.update(**new_kwargs)
        self.kwargs = kwargs

    def __call__(self):
        library = self.library
        model = self.model
        kwargs = deepcopy(self.kwargs)
        kwargs.update(**kwargs.pop("kwargs", {}))
        data = self.data
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        devices = range(torch.cuda.device_count())
        if str(type(model)).startswith("art.") and hasattr(model, "model"):
            model = model.model
        if "optimizer" in kwargs:
            optimizer = TorchOptimizer(**kwargs.pop("optimizer"))(model)
            kwargs.update({"optimizer": optimizer})
        else:
            optimizer = torch.optim.Adam(model.parameters())
            kwargs.update({"optimizer": optimizer})
        if "criterion" in kwargs:
            criterion = TorchCriterion(**kwargs.pop("criterion"))()
            kwargs.update({"loss": criterion})
        else:
            criterion = torch.nn.CrossEntropyLoss()
            kwargs.update({"loss": criterion})
        if "input_shape" not in kwargs:
            kwargs.update({"input_shape": data[0].shape[1:]})
        if "nb_classes" not in kwargs:
            if len(data[2].shape) == 1:
                kwargs.update({"nb_classes": len(np.unique(data[2]))})
            else:
                kwargs.update({"nb_classes": data[2].shape[1]})
        try:
            if hasattr(model, "to"):
                model.to(device)
            elif hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to(device)
        except Exception as e:
            if "CUDA out of memory" in str(e) and len(devices) > 0:
                device_number = devices[randint(0, len(devices) - 1)]
                device = f"cuda:{device_number}"
                logger.info(f"Out of memory error. Trying device {device}")
                model.to(device)
                for datum in data:
                    datum.to(device)
            else:
                raise e
        if library in torch_dict:
            kwargs.pop("library", None)
            model = torch_dict[library](model, **kwargs)
        else:
            raise NotImplementedError(f"Library {library} not implemented")
        return model
