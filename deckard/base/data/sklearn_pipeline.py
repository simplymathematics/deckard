import logging
from typing import Union
from omegaconf import OmegaConf
from hydra.utils import instantiate
from dataclasses import dataclass, field
from copy import deepcopy
from numpy import squeeze
from ..utils import my_hash

__all__ = ["SklearnDataPipelineStage", "SklearnDataPipeline"]
logger = logging.getLogger(__name__)


@dataclass
class SklearnDataPipelineStage:
    name: str
    kwargs: dict = field(default_factory=dict)
    y: bool = False
    X_only = False

    def __init__(self, name, y=False, X_only=False, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.y = y
        self.X_only = X_only

    def __hash__(self):
        return int(my_hash(self), 16)

    def __call__(self, X_train, X_test, y_train, y_test):
        name = self.kwargs.pop("_target_", self.name)
        dict_ = {"_target_": name}
        dict_.update(**self.kwargs)
        while "kwargs" in dict_:
            dict_.update(**dict_.pop("kwargs"))
        obj = instantiate(dict_)
        X_tr = squeeze(X_train)
        X_te = squeeze(X_test)
        y_tr = squeeze(y_train)
        y_te = squeeze(y_test)
        if self.y is False:
            if not self.X_only:
                X_tr_transformed = obj.fit_transform(X_tr, y_tr)
                X_te_transformed = obj.transform(X_te, y_te)
            else:
                X_tr_transformed = obj.fit_transform(X_tr)
                X_te_transformed = obj.transform(X_te)
            y_tr_transformed = y_tr
            y_te_transformed = y_te
        else:
            y_tr_transformed = obj.fit_transform(y_tr)
            y_te_transformed = obj.transform(y_te)
        return X_tr_transformed, X_te_transformed, y_tr_transformed, y_te_transformed


@dataclass
class SklearnDataPipeline:
    pipeline: Union[dict, None] = field(default_factory=dict)

    def __init__(self, **kwargs):
        pipe = kwargs.pop("pipeline", {})
        pipe.update(**kwargs)
        for stage in pipe:
            pipe[stage] = OmegaConf.to_container(
                OmegaConf.create(pipe[stage]),
                resolve=True,
            )
            name = pipe[stage].pop("name", stage)
            logger.debug(f"Instantiating {name} with kwargs {pipe[stage]}")
            pipe[stage] = SklearnDataPipelineStage(name, **pipe[stage])
        self.pipeline = pipe

    def __getitem__(self, key):
        return self.pipeline[key]

    def __len__(self):
        return len(self.pipeline)

    def __hash__(self):
        return int(my_hash(self), 16)

    def __iter__(self):
        return iter(self.pipeline)

    def __call__(self, X_train, X_test, y_train, y_test):
        logger.debug(
            "Calling SklearnDataPipeline with pipeline={}".format(self.pipeline),
        )
        pipeline = deepcopy(self.pipeline)
        for stage in pipeline:
            transformer = pipeline[stage]
            X_train, X_test, y_train, y_test = transformer(
                X_train,
                X_test,
                y_train,
                y_test,
            )
        return [X_train, X_test, y_train, y_test]
