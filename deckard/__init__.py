import logging
import os
from pathlib import Path
import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning,ConvergenceWarning
from optuna.exceptions import ExperimentalWarning
import hashlib
from omegaconf import OmegaConf
import json
import yaml


from .data import DataConfig
from .model import ModelConfig
from .model.defend import DefenseConfig
from .attack import AttackConfig
from .experiment import ExperimentConfig
from .file import FileConfig
from .score import ScorerDictConfig
from .utils import *
from .plot import YellowbrickConfigList, YellowbrickPlotConfig


DECKARD_CONFIG_DIR = os.environ.get("DECKARD_CONFIG_DIR", "config")
DECKARD_DEFAULT_CONFIG_FILE = os.environ.get(
    "DECKARD_DEFAULT_CONFIG_FILE",
    "default_experiment.yaml",
)
   
def _load_yaml_file(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def _file_resolver(arg: str):
    """
    Usage:
      ${file:search/rf.yaml:model_search}
      ${file:./configs/search/rf.yaml:model_search.subkey}
      ${file:/abs/path/to/file.yaml}       -> returns whole file
    """
    if not arg:
        raise ValueError("file resolver requires an argument like 'path/to/file.yaml[:key]'")

    # split into path and optional key (only first ':' splits, keys may contain '.')
    if ":" in arg:
        path_part, key_part = arg.split(":", 1)
        key_part = key_part.strip()
    else:
        path_part, key_part = arg, None
    path = Path(DECKARD_CONFIG_DIR, path_part)
    if not path.exists():
        raise FileNotFoundError(f"file resolver: file not found: {path_part} in working dir {os.getcwd()}")
    
    data = _load_yaml_file(path)
    # if user requested a nested key, walk the dict using dot-splitting
    if key_part:
        parts = key_part.split(".")
        cur = data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                raise KeyError(f"file resolver: key '{key_part}' not found in {path}")
        data = cur
    data = OmegaConf.create(data)
    # Return as an OmegaConf node so structured content is preserved
    return data

# Register resolver with OmegaConf (Hydra will pick up this plugin module automatically)
OmegaConf.register_new_resolver("file", _file_resolver, replace=True, use_cache=True)


def _merge_resolver(*args):
    """
    Merge multiple OmegaConf or dict objects into a single OmegaConf dict.
    Usage:
      ${merge:${file:search/rf.yaml:model_search}, ${file:search/class_labels.yaml:model_search}}
    """
    merged = OmegaConf.create()
    for arg in args:
        # Resolve any interpolations
        obj = OmegaConf.to_container(OmegaConf.create(arg), resolve=True)
        merged = OmegaConf.merge(merged, obj)
    return OmegaConf.create(merged)

OmegaConf.register_new_resolver("merge", _merge_resolver, replace=True)
def _normalize(value, root):
        target = value

        # Allow key-path lookup form: ${hash:data}
        if isinstance(value, str) and root is not None:
            selected = OmegaConf.select(root, value, default=None)
            if selected is not None:
                target = selected

        # Resolve OmegaConf nodes to plain Python containers
        if OmegaConf.is_config(target):
            return OmegaConf.to_container(target, resolve=True)
        return target
    
def _hash_conf(*values, _root_=None):
    """
    Supports:
      - ${hash:${data}}
      - ${hash:data_string}
      - ${hash:${data},${model}}
      - ${hash:data_string,model_string}
    """
    if not values:
        target = _root_
    elif len(values) == 1:
        target = _normalize(values[0], root=_root_)
    else:
        target = [_normalize(v, root=_root_) for v in values]

    s = json.dumps(target, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

OmegaConf.register_new_resolver("hash", _hash_conf, replace=True, use_cache=False)



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


