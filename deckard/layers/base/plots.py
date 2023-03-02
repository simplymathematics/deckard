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
class PlotsConfig:
    balance: Union[str, None] = None
    classification: Union[str, None] = None
    confusion: Union[str, None] = None
    correlation: Union[str, None] = None
    radviz: Union[str, None] = None
    rank: Union[str, None] = None
    roc_auc: Union[str, None] = None
    error: Union[str, None] = None
    residuals : Union[str, None] = None
    alphas: Union[str, None] = None
    silhouette: Union[str, None] = None
    elbow: Union[str, None] = None
    intercluster: Union[str, None] = None
    validation: Union[str, None] = None
    learning : Union[str, None] = None
    cross_validation : Union[str, None] = None
    feature_importances : Union[str, None] = None
    recursive : Union[str, None] = None
    dropping_curve : Union[str, None] = None
    rank1d: Union[str, None] = None
    rank2d: Union[str, None] = None
    parallel: Union[str, None] = None
    manifold: Union[str, None] = None
    filetype : str = "pdf"