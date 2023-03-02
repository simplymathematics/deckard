import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import hydra
import numpy as np
import yaml
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from deckard.base.hashable import my_hash
from deckard.layers.parse import parse

@dataclass
class AttackConfig:
    name : str = "art.attacks.evasion.FastGradientMethod"
    init : dict = field(default_factory=dict)
    generate : dict = field(default_factory=dict)
    filename : Union[str, None] = None
    path: Union[str, None] = None
    filetype: Union[str, None] = None