import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union
import pandas as pd
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor
from art.utils import to_categorical
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
        params = self.kwargs
        name = params.pop("_target_", self.name)
        dict_ = {"_target_": name}
        dict_.update(**params)
        if hasattr(model, "parameters"):
            dict_.update({"params": model.parameters()})
        else:  # pragma: no cover
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
        self.kwargs = kwargs

    def __call__(self):
        library = self.library
        model = self.model
        kwargs = deepcopy(self.kwargs)
        kwargs.update(**kwargs.pop("kwargs", {}))
        data = self.data
        optimizer = TorchOptimizer(
            **kwargs.pop("optimizer", {"name": "torch.optim.Adam"})
        )(model)
        kwargs.update({"optimizer": optimizer})
        criterion = TorchCriterion(
            **kwargs.pop("criterion", {"name": "torch.nn.CrossEntropyLoss"})
        )()
        kwargs.update({"loss": criterion})
        if "input_shape" not in kwargs:
            kwargs.update({"input_shape": data[0].shape[1:]})
        if "nb_classes" not in kwargs:
            if len(data[2].shape) == 1:  # pragma: no cover
                data[2] = to_categorical(data[2])
                data[3] = to_categorical(data[3])
            kwargs.update({"nb_classes": data[2].shape[1]})
        if library in torch_dict and not isinstance(model, torch_dict[library]):
            kwargs.pop("library", None)
            model = torch_dict[library](model, **kwargs)
        else:  # pragma: no cover
            raise NotImplementedError(f"Library {library} not implemented")
        return model
