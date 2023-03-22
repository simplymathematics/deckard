from dataclasses import dataclass, field

from deckard.layers.base.data import DataConfig
from deckard.layers.base.model import ModelConfig
from deckard.layers.base.attack import AttackConfig
from deckard.layers.base.plots import PlotsConfig


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scorers: dict = field(default_factory=dict)
    attack: AttackConfig = field(default_factory=AttackConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    files: dict = field(default_factory=dict)
