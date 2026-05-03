from dataclasses import dataclass
from typing import Union

from .base import ModelConfig
from ..data.anjana import AnjanaDataConfig
from ..utils import load_class


@dataclass(eq=False)
class AnjanaModelConfig(ModelConfig):
    """ANJANA-aware model config for data anonymization scoring."""

    data: Union[AnjanaDataConfig, None] = None

    def __post_init__(self):
        if isinstance(self.scorer, str) and self.scorer.lower() in {
            "auto",
            "default",
        }:
            self.scorer = load_class(
                "deckard.score.anjana.DefaultAnjanaModelScoreConfig",
            )
        super().__post_init__()
