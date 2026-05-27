"""Anjana model plugin implementation."""

from dataclasses import dataclass
from typing import Union

from ...model.base import ModelConfig
from ...utils import is_default_config_value, load_class, safe_store
from .data import AnjanaDataConfig


class AnjanaModelInitMixin:
    """Reusable initialization behavior for ANJANA model configs.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    # Declared for static analyzers; concrete dataclass provides these fields.
    data: Union[AnjanaDataConfig, None]

    def _before_post_init(self) -> None:
        if self.data is not None:
            self.data = self.coerce_component(
                self.data,
                AnjanaDataConfig,
                default_target="deckard.plugins.anjana.AnjanaDataConfig",
            )


@dataclass(eq=False, kw_only=True)
class AnjanaModelConfig(AnjanaModelInitMixin, ModelConfig):
    """ANJANA-aware model config for data anonymization scoring.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    data: Union[AnjanaDataConfig, None] = None

    def __post_init__(self):
        # Support test patterns that call __post_init__ directly on bare instances.
        self._before_post_init()
        if is_default_config_value(self.scorer, include_best=False):
            self.scorer = load_class(
                "deckard.plugins.anjana.score.DefaultAnjanaScorerDictConfig",
            )
        super().__post_init__()


ANJANA_MODEL = {
    "model_type": "sklearn.linear_model.LogisticRegression",
    "classifier": True,
    "model_params": {
        "max_iter": 10,
    },
    "data": "${data}",
    "alias": "anjana",
    "_target_": "deckard.plugins.anjana.model.AnjanaModelConfig",
}

safe_store(group="model", name="anjana", node=ANJANA_MODEL)
safe_store(group="search/models", name="anjana", node=ANJANA_MODEL)


__all__ = ["AnjanaModelConfig"]
