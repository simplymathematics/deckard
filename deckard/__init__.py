import logging
import os
import tempfile
import warnings

from sklearn.exceptions import UndefinedMetricWarning

from .base import *  # noqa: F401, F403
from .base import Data as Data
from .base import Model as Model
from .base import Attack as Attack
from .base import Experiment as Experiment
from .base import FileConfig as FileConfig
from .base import ScorerDict as ScorerDict

# from deckard import layers  # noqa: F401

# Semantic Version
__version__ = "0.630"

# pylint: disable=C0103

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
            "class": "logging.FileHandler",
            "filename": os.path.join(tempfile.gettempdir(), "deckard.log"),
            "mode": "a",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.DEBUG,
        },
    },
    "loggers": {
        "deckard": {"handlers": ["default"]},
        "tests": {"handlers": ["test"], "level": "DEBUG", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
