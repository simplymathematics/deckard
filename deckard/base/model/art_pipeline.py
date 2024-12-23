import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Dict, Union
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from art.estimators import BaseEstimator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
from random import randint
from .keras_models import KerasInitializer, keras_dict  # noqa F401
from .tensorflow_models import (  # noqa F401
    TensorflowV1Initializer,
    TensorflowV2Initializer,
    tensorflow_dict,
)
from .torch_models import TorchInitializer, torch_dict
from .sklearn_pipeline import sklearn_dict
from ..utils import my_hash

__all__ = ["ArtPipelineStage", "ArtModelPipeline"]
logger = logging.getLogger(__name__)

non_sklearn_models = {**torch_dict, **keras_dict, **tensorflow_dict}

all_models = {**sklearn_dict, **non_sklearn_models}
supported_models = all_models.keys()

__all__ = ["ArtPipelineStage", "ArtPipeline", "ArtInitializer"]


@dataclass
class ArtPipelineStage:
    name: Union[str, None] = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name=None, **kwargs):
        self.name = name
        kwargs.update(**kwargs.pop("kwargs", {}))
        self.kwargs = kwargs


@dataclass
class ArtSklearnInitializer:
    library: str = None
    data: list = None
    model: object = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, model: object, library: str, data=None, **kwargs):
        assert (
            library in supported_models
        ), f"library must be one of {supported_models}. Got {library}"
        self.library = library
        self.data = data
        try:
            check_is_fitted(model)
        except NotFittedError:
            raise ValueError("Model must be fitted before being passed to ART")
        self.model = model
        self.kwargs = kwargs

    def __call__(self):
        library = self.library
        model = self.model
        kwargs = self.kwargs
        if library in sklearn_dict and "art." not in str(type(model)):
            est = sklearn_dict[library]
            model = est(model, **kwargs)
        return model


# @dataclass
# class ArtInitializer:
#     library: str = None
#     data: list = None
#     model: object = None
#     kwargs: dict = field(default_factory=dict)

#     def __init__(self, model: object, library: str, data=None, **kwargs):
#         assert (
#             library in supported_models
#         ), f"library must be one of {supported_models}. Got {library}"
#         self.library = library
#         self.data = data
#         self.model = model
#         self.kwargs = kwargs

#     def __call__(self, model =None):
#         library = self.library
#         data = self.data
#         if model is None:
#             model = self.model
#         kwargs = self.kwargs
#         if "torch" in str(library) and not isinstance(
#             model,
#             tuple(torch_dict.values()),
#         ):
#             import torch
#             device_type = "gpu" if torch.cuda.is_available() else "cpu"
#             if device_type == "gpu":
#                 number_of_devices = torch.cuda.device_count()
#                 num = randint(0, number_of_devices - 1)
#                 device = torch.device(f"cuda:{num}")
#                 if isinstance(data[0][0], np.ndarray):
#                     data = [torch.from_numpy(d).to(device) for d in data]
#                 data = [d.to(device) for d in data]
#                 model.to(device)
#                 logger.debug(f"Model moved to GPU: {device}")
#             else:
#                 device = torch.device("cpu")
#             model = TorchInitializer(
#                 data=data,
#                 model=model,
#                 library=library,
#                 device_type=device,
#                 **kwargs,
#             )()
#         elif "keras" in str(library) and not isinstance(
#             model,
#             tuple(keras_dict.values()),
#         ):  # pragma: no cover
#             raise NotImplementedError("Keras not implemented yet")
#             # try:
#             #     model = KerasInitializer(
#             #         data=data, model=model, library=library, **kwargs
#             #     )()
#             # except ValueError as e:
#             #     if "disable eager execution" in str(e):
#             #         import tensorflow as tf

#             #         tf.compat.v1.disable_eager_execution()
#             #         if str(type(model)).startswith("<class 'art."):
#             #             model = model.model
#             #         model = KerasInitializer(
#             #             data=data, model=model, library=library, **kwargs
#             #         )()
#             #     else:
#             #         raise e
#         elif (
#             "sklearn" in str(library)
#             or library is None
#             and not isinstance(model, tuple(sklearn_dict.values()))
#         ):
#             try:
#                 check_is_fitted(model)
#             except NotFittedError:
#                 raise ValueError("Model must be fitted before being passed to ART")
#             if library in sklearn_dict and "art." not in str(type(model)):
#                 est = sklearn_dict[library]
#                 model = est(model, **kwargs)
#         elif library in [
#             "tf2",
#             "tensorflowv2",
#             "tensorflow",
#             "tf",
#             "tfv2",
#         ] and not isinstance(model, tuple(tensorflow_dict.values())):
#             model = TensorflowV2Initializer(
#                 data=data,
#                 model=model,
#                 library=library,
#                 **kwargs,
#             )()
#         elif library in ["tf1", "tensorflowv1", "tfv1"] and not isinstance(
#             model,
#             tuple(tensorflow_dict.values()),
#         ):  # pragma: no cover
#             raise NotImplementedError("Tensorflow V1 not implemented yet")
#             # model = TensorflowV1Initializer(
#             #     data=data, model=model, library=library, **kwargs
#             # )()
#         elif library in supported_models and isinstance(
#             model,
#             tuple(all_models.values()),
#         ):
#             pass
#         else:  # pragma: no cover
#             raise ValueError(
#                 f"library must be one of {supported_models}. Got {library}",
#             )
#         assert hasattr(
#             model,
#             "fit",
#         ), f"model must have a fit method. Got type {type(model)}"
#         return model


@dataclass
class ArtKerasInitializer:
    library: str = None
    data: list = None
    model: object = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, model: object, library: str, data=None, **kwargs):
        assert (
            library in supported_models
        ), f"library must be one of {supported_models}. Got {library}"
        self.library = library
        self.data = data
        self.model = model
        self.kwargs = kwargs

    def __call__(self):
        library = self.library
        model = self.model
        kwargs = self.kwargs
        if "keras" in str(library) and not isinstance(
            model,
            tuple(keras_dict.values()),
        ):
            try:
                model = KerasInitializer(
                    data=self.data,
                    model=model,
                    library=library,
                    **kwargs,
                )()
            except ValueError as e:
                if "disable eager execution" in str(e):
                    import tensorflow as tf

                    tf.compat.v1.disable_eager_execution()
                    if str(type(model)).startswith("<class 'art."):
                        model = model.model
                    model = KerasInitializer(
                        data=self.data,
                        model=model,
                        library=library,
                        **kwargs,
                    )()
                else:
                    raise e
        return model


@dataclass
class ArtTF2Initializer:
    library: str = None
    data: list = None
    model: object = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, model: object, library: str, data=None, **kwargs):
        assert (
            library in supported_models
        ), f"library must be one of {supported_models}. Got {library}"
        self.library = library
        self.data = data
        self.model = model
        self.kwargs = kwargs

    def __call__(self):
        library = self.library
        model = self.model
        kwargs = self.kwargs
        if library in [
            "tf2",
            "tensorflowv2",
            "tensorflow",
            "tf",
            "tfv2",
        ] and not isinstance(model, tuple(tensorflow_dict.values())):
            model = TensorflowV2Initializer(
                data=self.data,
                model=model,
                library=library,
                **kwargs,
            )()
        return model


@dataclass
class ArtPytorchInitializer:
    library: str = None
    data: list = None
    model: object = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, model: object, library: str, data=None, **kwargs):
        assert (
            library in supported_models
        ), f"library must be one of {supported_models}. Got {library}"
        self.library = library
        self.data = data
        self.model = model
        self.kwargs = kwargs

    def __call__(self):
        library = self.library
        model = self.model
        kwargs = self.kwargs
        data = self.data
        if "torch" in str(library) and not isinstance(
            model,
            tuple(torch_dict.values()),
        ):
            import torch

            device_type = "gpu" if torch.cuda.is_available() else "cpu"
            if device_type == "gpu":
                number_of_devices = torch.cuda.device_count()
                num = randint(0, number_of_devices - 1)
                device = torch.device(f"cuda:{num}")
                if isinstance(data[0][0], np.ndarray):
                    data = [torch.from_numpy(d).to(device) for d in data]
                data = [d.to(device) for d in data]
                model.to(device)
                logger.debug(f"Model moved to GPU: {device}")
            else:
                device = torch.device("cpu")
            model = TorchInitializer(
                data=data,
                model=model,
                library=library,
                device_type=device,
                **kwargs,
            )()
        return model


@dataclass
class ArtInitializer:
    library: str = None
    data: list = None
    model: object = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, model: object, library: str, data=None, **kwargs):
        assert (
            library in supported_models
        ), f"library must be one of {supported_models}. Got {library}"
        self.library = library
        self.data = data

        self.model = model
        self.kwargs = kwargs

    def __call__(self):
        if self.library in sklearn_dict:
            try:
                check_is_fitted(self.model)
            except NotFittedError:
                raise ValueError("Model must be fitted before being passed to ART")
            model = ArtSklearnInitializer(
                model=self.model,
                library=self.library,
                data=self.data,
                **self.kwargs,
            )()
        elif self.library in keras_dict:
            model = ArtKerasInitializer(
                model=self.model,
                library=self.library,
                data=self.data,
                **self.kwargs,
            )()
        elif self.library in tensorflow_dict:
            model = ArtTF2Initializer(
                model=self.model,
                library=self.library,
                data=self.data,
                **self.kwargs,
            )()
        elif self.library in torch_dict:
            model = ArtPytorchInitializer(
                model=self.model,
                library=self.library,
                data=self.data,
                **self.kwargs,
            )()
        return model


@dataclass
class ArtPipeline:
    library: str = None
    pipeline: Dict[str, ArtPipelineStage] = field(default_factory=dict)
    name: str = None

    def __init__(self, library, name: str = None, **kwargs):
        self.library = library
        pipeline = deepcopy(kwargs.pop("pipeline", {}))
        pipeline.update(**kwargs)
        pipeline.pop("data", None)
        pipeline.pop("model", None)
        for stage in pipeline:
            if isinstance(pipeline[stage], DictConfig):
                pipeline[stage] = OmegaConf.to_container(pipeline[stage], resolve=True)
            if isinstance(pipeline[stage], dict):
                pipeline[stage].update(**pipeline[stage].pop("kwargs", {}))
                name = pipeline[stage].pop("name", stage)
                params = pipeline[stage]
                params.pop("name", None)
                pipeline[stage] = ArtPipelineStage(name, **params)
            elif isinstance(pipeline[stage], type(None)):
                pipeline[stage] = ArtPipelineStage(name=stage)
        self.pipeline = pipeline
        self.name = kwargs.pop("name", my_hash(vars(self)))

    def __len__(self):
        return len(self.pipeline) if self.pipeline is not None else 0

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(self, model: object, data: list) -> BaseEstimator:
        if "initialize" in self.pipeline:
            params = asdict(self.pipeline["initialize"])
            name = params.pop("name", None)
            kwargs = params.pop("kwargs", {})
        else:
            name = self.library
            kwargs = {}
        pre_def = []
        post_def = []
        library = self.library
        if "preprocessor" in self.pipeline:
            params = asdict(self.pipeline["preprocessor"])
            name = params.pop("name", None)
            sub_kwargs = params.pop("kwargs", {})
            sub_kwargs.update(sub_kwargs.pop("params", {}))
            config = {
                "_target_": name,
            }
            config.update(**sub_kwargs)
            obj = instantiate(config)
            pre_def.append(obj)
            kwargs.update({"preprocessing_defences": pre_def})
        if "postprocessor" in self.pipeline:
            params = asdict(self.pipeline["postprocessor"])
            name = params.pop("name")
            config = {
                "_target_": name,
            }
            sub_kwargs = params.pop("kwargs", {})
            sub_kwargs.update(sub_kwargs.pop("params", {}))
            config = {
                "_target_": name,
            }
            config.update(**sub_kwargs)
            obj = instantiate(config)
            post_def.append(obj)
            kwargs.update({"postprocessing_defences": post_def})
        if isinstance(model, tuple(list(sklearn_dict.values()))):
            try:
                check_is_fitted(model)
                print("Model is fitted")
            except NotFittedError:
                print("Model not fitted")
                model.fit(data[0], data[2])
            print(f"Type of model: {type(model)}")
            input("Press Enter to continue...")
        model = ArtInitializer(model=model, data=data, **kwargs, library=library)()
        if "transformer" in self.pipeline:  # pragma: no cover
            raise NotImplementedError("Transformation defences not implemented yet")
            # name, sub_kwargs = self.pipeline["transformer"]()
            # config = {
            #     "_target_": name,
            # }
            # config.update(**sub_kwargs)
            # model = obj(model)
        if "trainer" in self.pipeline:  # pragma: no cover
            # name, sub_kwargs = self.pipeline["trainer"]()
            # assert "attack" in sub_kwargs, "Attack must be specified if the adversarial training defence is chosen."
            # attack = sub_kwargs.pop("attack")
            # if isinstance(attack, DictConfig):
            #     attack = OmegaConf.to_container(attack, resolve=True)
            #     attack = instantiate(attack)
            # elif is_dataclass(attack):
            #     attack = asdict(attack)
            #     attack['_target_'] = attack.pop('name')
            #     attack = instantiate(attack, model)
            # elif isinstance(attack, dict):
            #     attack['_target_'] = attack.pop('name')
            #     attack = instantiate(attack, model)
            # else:
            #     assert "art.attacks" in str(type(attack)), f"Attack must be an art attack. Got {type(attack)}"
            # if name == "art.defences.trainer.AdversarialTrainer":
            #     from art.defences.trainer.adversarial_trainer import AdversarialTrainer
            #     model = AdversarialTrainer(classifier=model, attacks = attack, **sub_kwargs)
            # else:
            raise NotImplementedError("Training Defense not implemented yet")
        return model
