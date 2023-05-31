import logging
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Dict, Union
from art.estimators import BaseEstimator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .keras_models import KerasInitializer, keras_dict
from .tensorflow_models import (
    TensorflowV1Initializer,
    TensorflowV2Initializer,
    tensorflow_dict,
)
from .torch_models import TorchInitializer, torch_dict
from .sklearn_pipeline import SklearnModelInitializer, sklearn_dict
from ..utils import my_hash

__all__ = ["ArtPipelineStage", "ArtModelPipeline"]
logger = logging.getLogger(__name__)


all_models = {**sklearn_dict, **torch_dict, **keras_dict, **tensorflow_dict}
supported_models = all_models.keys()

__all__ = ["ArtPipelineStage", "ArtPipeline", "ArtInitializer"]


@dataclass
class ArtPipelineStage:
    name: Union[str, None] = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name=None, **kwargs):
        logger.info(f"Creating pipeline stage: {name} kwargs: {kwargs}")
        self.name = name
        self.kwargs = kwargs

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(self):
        return self.name, self.kwargs


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

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(self):
        library = self.library
        data = self.data
        model = self.model
        kwargs = self.kwargs
        if "torch" in str(library):
            model = TorchInitializer(
                data=data, model=model, library=library, **kwargs
            )()
        elif "keras" in str(library):
            try:
                model = KerasInitializer(
                    data=data, model=model, library=library, **kwargs
                )()
            except ValueError as e:
                if "disable eager execution" in str(e):
                    import tensorflow as tf

                    tf.compat.v1.disable_eager_execution()
                    if str(type(model)).startswith("<class 'art."):
                        model = model.model
                    model = KerasInitializer(
                        data=data, model=model, library=library, **kwargs
                    )()
                else:
                    raise e
        elif "sklearn" in str(library) or library is None:
            model = SklearnModelInitializer(
                data=data, model=model, library=library, **kwargs
            )()
        elif library in ["tf2", "tensorflowv2", "tensorflow", "tf", "tfv2"]:
            model = TensorflowV2Initializer(
                data=data, model=model, library=library, **kwargs
            )()
        elif library in ["tf1", "tensorflowv1", "tfv1"]:
            model = TensorflowV1Initializer(
                data=data, model=model, library=library, **kwargs
            )()
        else:
            raise ValueError(
                f"library must be one of {supported_models}. Got {library}",
            )
        assert hasattr(
            model,
            "fit",
        ), f"model must have a fit method. Got type {type(model)}"
        return model


@dataclass
class ArtPipeline:
    library: str = None
    pipeline: Dict[str, ArtPipelineStage] = field(default_factory=dict)
    name: str = None

    def __init__(self, library, name: str = None, **kwargs):
        self.library = library
        pipeline = deepcopy(kwargs.pop("pipeline", {}))
        pipeline.update(kwargs)
        pipeline.pop("data", None)
        pipeline.pop("model", None)
        for stage in pipeline:
            if isinstance(pipeline[stage], DictConfig):
                pipeline[stage] = OmegaConf.to_container(pipeline[stage])
            elif is_dataclass(pipeline[stage]):
                pipeline[stage] = asdict(pipeline[stage])
            else:
                if not isinstance(
                    pipeline[stage],
                    dict,
                ):
                    pipeline[stage] = {**pipeline[stage]} if pipeline[stage] is not None else {}
            while "kwargs" in pipeline[stage]:
                pipeline[stage].update(**pipeline[stage].pop("kwargs"))
            while "params" in pipeline[stage]:
                pipeline[stage].update(**pipeline[stage].pop("params"))
            name = pipeline[stage].pop("name", stage)
            params = pipeline[stage]
            params.pop("name", None)
            pipeline[stage] = ArtPipelineStage(name, **params)
        self.pipeline = pipeline
        self.name = kwargs.pop("name", my_hash(vars(self)))

    def __len__(self):
        if self.pipeline is not None:
            return len(self.pipeline)
        return 0

    def __hash__(self):
        return int(my_hash(self), 16)

    # def __iter__(self):
    #     return iter(self.pipeline)

    def __call__(self, model: object, library: str = None, data=None) -> BaseEstimator:
        if "initialize" in self.pipeline:
            name, kwargs = self.pipeline["initialize"]()
        else:
            raise ValueError("Art Pipeline must have an initialize stage")
        pre_def = []
        post_def = []
        if data is None:
            data = model.data()
        if library is None:
            library = self.library
        else:
            assert (
                library in supported_models
            ), f"library must be one of {supported_models}. Got {library}"
        assert len(data) == 4, f"data must be a tuple of length 4. Got {data}"
        if "preprocessor" in self.pipeline:
            name, sub_kwargs = self.pipeline["preprocessor"]()
            config = {"_target_": name}
            config.update(**sub_kwargs)
            obj = instantiate(config)
            pre_def.append(obj)
            kwargs.update({"preprocessing_defences": pre_def})
        if "postprocessor" in self.pipeline:
            name, sub_kwargs = self.pipeline["postprocessor"]()
            config = {
                "_target_": name,
            }
            config.update(**sub_kwargs)
            obj = instantiate(config)
            post_def.append(obj)
            kwargs.update({"postprocessing_defences": post_def})
        model = ArtInitializer(model=model, data=data, **kwargs, library=library)()
        if "transformer" in self.pipeline:
            name, sub_kwargs = self.pipeline["transformer"]()
            config = {
                "_target_": name,
            }
            config.update(**sub_kwargs)
            model = obj(model)
        if "trainer" in self.pipeline:
            raise NotImplementedError("Training Defense not implemented yet")
        return model
