from dataclasses import dataclass
from typing import Union

from .base import ModelConfig
from ..data.anjana import AnjanaDataConfig
from ..utils import is_default_config_value, load_class
@dataclass(eq=False)
class AnjanaModelConfig(ModelConfig):
    """ANJANA-aware model config for data anonymization scoring."""

    data: Union[AnjanaDataConfig, None] = None

    def __post_init__(self):
        if is_default_config_value(self.scorer, include_best=False):
            self.scorer = load_class(
                "deckard.score.anjana.DefaultAnjanaModelScoreConfig",
            )
        super().__post_init__()
