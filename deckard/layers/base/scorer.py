from dataclasses import dataclass, field
from typing import List, Union, Tuple, Any, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import yaml
from pathlib import Path
from deckard.base.hashable import my_hash

@dataclass
class ScorerConfig:
    name : str = "sklearn.metrics.accuracy_score"
    params : dict = field(default_factory=dict)
    alias : str = "accuracy"

@dataclass
class ScorerDictConfig:
    scorers : dict = field(default_factory=dict)