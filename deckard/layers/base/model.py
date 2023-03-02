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
class ArtPipelineConfig:
    stages: dict = field(default_factory=dict)

@dataclass
class SklearnPipelineConfig:
    stages: dict = field(default_factory=dict)

@dataclass
class ModelConfig:
    library : str = "sklearn"
    name : str = "sklearn.linear_model.SGDClassifier"
    init : dict = field(default_factory=dict)
    fit : dict = field(default_factory=dict)
    sklearn_pipeline : dict = field(default_factory=dict)
    art_pipeline : dict = field(default_factory=dict)
    filename : Union[str, None] = None
    path : Union[str, None] = None  
    filetype : Union[str, None] = None 