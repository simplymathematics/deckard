

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import numpy as np
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from deckard.base.hashable import my_hash
from deckard.layers.base.data import DataConfig
from deckard.layers.base.model import ModelConfig
from deckard.layers.base.attack import AttackConfig
from deckard.layers.base.plots import PlotsConfig
from deckard.layers.parse import parse


@dataclass
class ExperimentConfig:
    data : DataConfig = field(default_factory=DataConfig)
    model : ModelConfig = field(default_factory=ModelConfig)
    scorers : dict = field(default_factory = dict)
    attack : AttackConfig = field(default_factory=AttackConfig)
    plots : PlotsConfig = field(default_factory=PlotsConfig)
    files : dict = field(default_factory=dict)