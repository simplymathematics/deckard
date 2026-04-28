import logging
import os
from pathlib import Path
import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning,ConvergenceWarning
from optuna.exceptions import ExperimentalWarning

from .data import DataConfig
from .model import ModelConfig
from .model.defend import DefenseConfig
from .attack import AttackConfig
from .experiment import ExperimentConfig
from .file import FileConfig
from .score import ScorerDictConfig
from .utils import *



"""
deckard
=======

Main package initializer for the deckard adversarial machine learning framework.

This module configures logging, suppresses common warnings, and exposes the primary
public API classes for data, model, defense, attack, experiment, file, and scoring
configurations.

Logging Configuration
---------------------
Logging is configured via :func:`logging.config.dictConfig` with the following setup:

- **File handler** (``default``):
    - Uses :class:`logging.handlers.RotatingFileHandler`.
    - Writes to ``deckard.log`` in the current working directory.
    - Log level: ``INFO``.
    - Max file size: 10 MB, with up to 5 rotating backups.

- **Stream handler** (``stream``):
    - Writes to stdout via :class:`logging.StreamHandler`.
    - Log level: ``DEBUG``.

- **Logger ``deckard``**:
    - Handlers: ``["default"]`` (file only).
    - Level: ``INFO``.
    - ``propagate=True``.

- **Logger ``tests``**:
    - Handlers: ``["stream"]`` (stdout only).
    - Level: ``DEBUG``.
    - ``propagate=True``.

.. warning::
    The ``propagate=True`` setting on child loggers (e.g., ``deckard.attack``,
    ``deckard.model``, ``deckard.data``) causes their log records to bubble up
    to the root logger. If the root logger has a ``StreamHandler`` attached
    (which Python adds by default via :func:`logging.basicConfig` or third-party
    libraries such as ``optuna``, ``art``, or ``sklearn``), **all** ``logger.info``
    calls from submodules like ``deckard.attack``, ``deckard.model``, and
    ``deckard.data`` will appear on stdout in addition to being written to the
    log file.

    To suppress stdout output from submodule loggers, set ``propagate=False``
    on the ``deckard`` logger, or explicitly remove any ``StreamHandler`` from
    the root logger after calling :func:`logging.config.dictConfig`.

    Example fix::

        # Remove StreamHandlers from root logger to prevent stdout leakage:
        root = logging.getLogger()
        root.handlers = [h for h in root.handlers if not isinstance(h, logging.StreamHandler)]

Warning Suppression
-------------------
The following warning categories are suppressed globally:

- :class:`FutureWarning`
- :class:`DeprecationWarning`
- :class:`sklearn.exceptions.UndefinedMetricWarning`
- :class:`RuntimeWarning`
- :class:`sklearn.exceptions.ConvergenceWarning`
- :class:`optuna.exceptions.ExperimentalWarning`

Additionally, numpy divide-by-zero and invalid-value errors are silenced via
:func:`numpy.seterr`.

Public API
----------
.. autosummary::
    :toctree: _autosummary

    DataConfig
    ModelConfig
    AttackConfig
    ExperimentConfig
    DefenseConfig
    FileConfig
    ScorerDictConfig
"""

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
        "stream": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.INFO
        },
    },
    "loggers": {
        "deckard": {"handlers": ["default", "stream"],  "propagate": False},
        "tests": {"handlers": ["stream"], "level": "DEBUG", "propagate": True},
    },
}




logging.getLogger("art").setLevel(logging.WARNING)
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
np.seterr(divide='ignore', invalid='ignore')


