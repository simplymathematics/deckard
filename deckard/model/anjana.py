from dataclasses import dataclass
from typing import Union

from .base import ModelConfig
from ..data.anjana import AnjanaDataConfig
from ..utils import is_default_config_value, load_class


@dataclass(eq=False)
class AnjanaModelConfig(ModelConfig):
    """ANJANA-aware model config for data anonymization scoring."""

    data: Union[AnjanaDataConfig, None] = None

    def _before_post_init(self) -> None:
        if self.data is not None:
            self.data = self.coerce_component(
                self.data,
                AnjanaDataConfig,
                default_target="deckard.data.anjana.AnjanaDataConfig",
            )

    def __post_init__(self):
        # Support test patterns that call __post_init__ directly on bare instances.
        self._before_post_init()
        if is_default_config_value(self.scorer, include_best=False):
            self.scorer = load_class(
                "deckard.score.anjana.DefaultAnjanaScoreConfig",
            )
        super().__post_init__()
