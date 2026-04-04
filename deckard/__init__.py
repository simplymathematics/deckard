import logging
import os
from pathlib import Path
import warnings
from sklearn.exceptions import UndefinedMetricWarning

from .data import DataConfig
from .model import ModelConfig
from .model.defend import DefenseConfig
from .attack import AttackConfig
from .experiment import ExperimentConfig
from .file import FileConfig
from .score import ScorerDictConfig
from .utils import *
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
logger = logging.getLogger(__name__)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "AttackConfig",
    "ExperimentConfig",
    "DefenseConfig",
    "FileConfig",
    "ScorerDictConfig",
]

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
        },
    },
    "handlers": {
        "default": {
            # Use RotatingFileHandler for log rotation
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(Path.cwd(), "deckard.log"),
            "formatter": "std",
            "level": logging.INFO,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB log file size limit
            "backupCount": 5,  # Keep up to 5 backup files
            "mode": "a",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.DEBUG,
        },
    },
    "loggers": {
        "deckard": {"handlers": ["default"], "level": "INFO", "propagate": True},
        "tests": {"handlers": ["test"], "level": "DEBUG", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)