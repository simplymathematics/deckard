import logging
from typing import Dict, Union
from dataclasses import dataclass, asdict, field, is_dataclass
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from copy import deepcopy
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from art.estimators.classification.scikitlearn import (
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnRandomForestClassifier,
    ScikitlearnSVC,
)
from art.estimators.regression.scikitlearn import (
    ScikitlearnDecisionTreeRegressor,
    ScikitlearnRegressor,
)


from ..utils import my_hash

__all__ = ["SklearnModelPipelineStage", "SklearnModelPipeline"]
logger = logging.getLogger(__name__)


classifier_dict = {
    "sklearn-svc": ScikitlearnSVC,
    "sklearn-logistic-regression": ScikitlearnLogisticRegression,
    "sklearn-random-forest": ScikitlearnRandomForestClassifier,
    "sklearn-extra-trees": ScikitlearnExtraTreesClassifier,
    "sklearn-decision-tree": ScikitlearnDecisionTreeClassifier,
    "sklearn-gradient-boosting": ScikitlearnGradientBoostingClassifier,
    "sklearn-bagging": ScikitlearnBaggingClassifier,
    "sklearn-adaboost": ScikitlearnAdaBoostClassifier,
    "sklearn": ScikitlearnClassifier,
}

regressor_dict = {
    "sklearn-decision-tree-regressor": ScikitlearnDecisionTreeRegressor,
    "sklearn-regressor": ScikitlearnRegressor,
}

sklearn_dict = {**classifier_dict, **regressor_dict}
sklearn_models = list(sklearn_dict.keys())


@dataclass
class SklearnModelPipelineStage:
    name: str
    stage_name: str = None
    kwargs: dict = field(default_factory=dict)

    def __init__(self, name, stage_name, **kwargs):
        logger.debug(
            f"Instantiating {self.__class__.__name__} with name={name} and kwargs={kwargs}",
        )
        self.name = name
        self.kwargs = kwargs
        self.stage_name = stage_name

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(self, model):
        logger.debug(
            f"Calling SklearnModelPipelineStage with name={self.name} and kwargs={self.kwargs}",
        )
        name = self.name
        kwargs = deepcopy(self.kwargs)
        stage_name = self.stage_name if self.stage_name is not None else name
        while "kwargs" in kwargs:
            kwargs.update(**kwargs.pop("kwargs"))
        if "art." in str(type(model)):
            assert isinstance(
                model.model,
                BaseEstimator,
            ), f"model must be a sklearn estimator. Got {type(model.model)}"
        else:
            assert isinstance(
                model,
                BaseEstimator,
            ), f"model must be a sklearn estimator. Got {type(model)}"
        if not isinstance(model, Pipeline):
            model = Pipeline([("model", model)])
        else:
            model.steps.insert(-2, [stage_name, model])
        assert isinstance(
            model,
            Pipeline,
        ), f"model must be a sklearn pipeline. Got {type(model)}"
        return model


@dataclass
class SklearnModelPipeline:
    pipeline: Dict[str, SklearnModelPipelineStage] = field(default_factory=dict)

    def __init__(self, **kwargs):
        logger.debug(f"Instantiating {self.__class__.__name__} with kwargs={kwargs}")
        pipe = {}
        while "kwargs" in kwargs:
            pipe.update(**kwargs.pop("kwargs"))
        pipe.update(**kwargs)
        for stage in pipe:
            if isinstance(pipe[stage], SklearnModelPipelineStage):
                pipe[stage] = asdict(pipe[stage])
            elif isinstance(pipe[stage], dict):
                pipe[stage] = pipe[stage]
            elif isinstance(pipe[stage], DictConfig):
                pipe[stage] = OmegaConf.to_container(pipe[stage], resolve=True)
            elif is_dataclass(pipe[stage]):
                pipe[stage] = asdict(pipe[stage])
            else:
                assert hasattr(
                    pipe[stage],
                    "transform",
                ), f"Pipeline stage must be a SklearnModelPipelineStage, dict, or have a transform method. Got {type(pipe[stage])}"
            if isinstance(pipe[stage], dict):
                params = deepcopy(pipe[stage])
                stage_name = params.pop("stage_name", stage)
                pipe[stage] = SklearnModelPipelineStage(params, stage_name=stage_name)
            elif hasattr(pipe[stage], "transform"):
                assert hasattr(
                    pipe[stage],
                    "fit",
                ), f"Pipeline stage must have a fit method. Got {type(pipe[stage])}"
            else:
                raise ValueError(
                    f"Pipeline stage must be a SklearnModelPipelineStage, dict, or have a transform method. Got {type(pipe[stage])}",
                )
        self.pipeline = pipe

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        if self.pipeline is not None:
            return len(self.pipeline)
        else:
            return 0

    def __hash__(self):
        return int(my_hash(self), 16)

    def __iter__(self):
        return iter(self.pipeline)

    def __call__(self, model):
        params = deepcopy(asdict(self))
        pipeline = params.pop("pipeline")
        for stage in pipeline:
            stage = pipeline[stage]
            if isinstance(stage, dict):
                stage["_target_"] = (
                    "deckard.base.model.sklearn_pipeline.SklearnModelPipelineStage"
                )
                stage = instantiate(stage)
                model = stage(model=model)
            elif isinstance(stage, DictConfig):
                stage = OmegaConf.to_container(stage, resolve=True)
                stage["_target_"] = (
                    "deckard.base.model.sklearn_pipeline.SklearnModelPipelineStage"
                )
                stage = instantiate(stage)
                model = stage(model=model)
            elif isinstance(stage, SklearnModelPipelineStage):
                model = stage(model=model)
            elif hasattr(stage, "fit"):
                if "art." in str(type(model)):
                    assert isinstance(
                        model.model,
                        BaseEstimator,
                    ), f"model must be a sklearn estimator. Got {type(model.model)}"
                else:
                    assert isinstance(
                        model,
                        BaseEstimator,
                    ), f"model must be a sklearn estimator. Got {type(model)}"
                if not isinstance(model, Pipeline) and "art." not in str(type(model)):
                    model = Pipeline([("model", model)])
                elif "art." in str(type(model)) and not isinstance(
                    model.model,
                    Pipeline,
                ):
                    model.model = Pipeline([("model", model.model)])
                elif "art." in str(type(model)) and isinstance(model.model, Pipeline):
                    model.model.steps.insert(-2, [stage, model.model])
                else:
                    model.steps.insert(-2, [stage, model])
                if "art." not in str(type(model)):
                    assert isinstance(
                        model,
                        Pipeline,
                    ), f"model must be a sklearn pipeline. Got {type(model)}"
                else:
                    assert isinstance(
                        model.model,
                        Pipeline,
                    ), f"model must be a sklearn pipeline. Got {type(model)}"
        return model


@dataclass
class SklearnModelInitializer:
    model: object = field(default_factory=None)
    library: str = field(default_factory="sklearn")
    pipeline: SklearnModelPipeline = field(default_factory=None)
    kwargs: Union[dict, None] = field(default_factory=dict)

    def __init__(self, data, model=None, library="sklearn", pipeline={}, **kwargs):
        self.data = data
        self.model = model
        self.library = library
        self.kwargs = kwargs
        if len(pipeline) > 0:
            self.pipeline = SklearnModelPipeline(**pipeline)
        else:
            self.pipeline = None

    def __call__(self):
        logger.debug(f"Initializing model {self.model} with kwargs {self.kwargs}")
        model = self.model
        if isinstance(model, BaseEstimator):
            pass
        elif isinstance(model, DictConfig):
            model = OmegaConf.to_container(model, resolve=True)
        elif isinstance(model, str):
            model = {"name": model, **self.kwargs}
        else:
            assert isinstance(
                model,
                dict,
            ), f"model must be a sklearn estimator, string, or dict. Got {type(model)}"
        if isinstance(model, dict):
            if "_target_" in model:
                name = model.pop("_target_")
            elif "name" in model:
                name = model.pop("name")
            else:
                raise ValueError(
                    f"model must have a name attribute. Got {model}",
                )
            model["target"] = name
        model = instantiate(model)
        if self.pipeline is not None:
            model = self.pipeline(model)
            assert isinstance(
                model,
                BaseEstimator,
            ), f"model must be a sklearn estimator. Got {type(model)}"
        assert hasattr(
            model,
            "fit",
        ), f"model must have a fit method. Got type {type(model)}"
        return model

    def __hash__(self):
        return int(my_hash(self), 16)
