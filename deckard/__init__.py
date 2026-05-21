"""Public package entrypoint for deckard.

This module configures warning filters, registers the OmegaConf resolvers used
throughout deckard configs, and re-exports the primary configuration objects
that make up the supported public API:

- ``DataConfig`` and related data configuration classes
- ``ModelConfig`` and ``DefenseConfig``
- ``AttackConfig``
- ``ExperimentConfig`` and ``SurvivalExperimentConfig``
- ``FileConfig``
- ``ScorerDictConfig``

Importing :mod:`deckard` is the supported top-level entrypoint for most users
who construct experiments from Python instead of the CLI.
"""

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf
from optuna.exceptions import ExperimentalWarning
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

# Install library warning filters before importing deckard submodules, since
# those imports can transitively import sklearn/art and emit warnings.
warnings.filterwarnings("ignore", module=r"^sklearn(\.|$)")
warnings.filterwarnings("ignore", module=r"^art(\.|$)")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"PyTorch not found\. Not importing DeepZ or Interval Bound Propagation functionality",
)

from .data import DataConfig  # noqa E402
from .model import ModelConfig  # noqa E402
from .model.defend import DefenseConfig  # noqa E402
from .attack import AttackConfig  # noqa E402
from .artifacts import ArtifactLoaderConfig  # noqa E402
from .detector import DetectorConfig  # noqa E402
from .experiment import ExperimentConfig  # noqa E402

try:
    from .experiment import SurvivalExperimentConfig  # noqa E402
except ImportError:  # pragma: no cover
    SurvivalExperimentConfig = None
from .file import FileConfig  # noqa E402
from .score import ScorerDictConfig  # noqa E402
from .utils import hash_conf_values  # noqa E402

# from .plot import YellowbrickConfigList, YellowbrickPlotConfig


DECKARD_CONFIG_DIR = os.environ.get("DECKARD_CONFIG_DIR", "config")
DECKARD_DEFAULT_CONFIG_FILE = os.environ.get(
    "DECKARD_DEFAULT_CONFIG_FILE",
    "default.yaml",
)

# those imports can transitively import sklearn/art and emit warnings.
warnings.filterwarnings("ignore", module=r"^sklearn(\.|$)")
warnings.filterwarnings("ignore", module=r"^art(\.|$)")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"PyTorch not found\. Not importing DeepZ or Interval Bound Propagation functionality",
)


def _load_yaml_file(path: Path):
    """Load a YAML file from disk and return the parsed Python object."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def _file_resolver(arg: str):
    """Resolve ``${file:...}`` OmegaConf interpolations relative to deckard config.

    Example:

    ```text
    ${file:search/rf.yaml:model_search}
    ${file:./configs/search/rf.yaml:model_search.subkey}
    ${file:/abs/path/to/file.yaml}
    ```
    """
    if not arg:
        raise ValueError(
            "file resolver requires an argument like 'path/to/file.yaml[:key]'",
        )

    # split into path and optional key (only first ':' splits, keys may contain '.')
    if ":" in arg:
        path_part, key_part = arg.split(":", 1)
        key_part = key_part.strip()
    else:
        path_part, key_part = arg, None
    path = Path(DECKARD_CONFIG_DIR, path_part)
    if not path.exists():
        raise FileNotFoundError(
            f"file resolver: file not found: {path_part} in working dir {os.getcwd()}",
        )

    data = _load_yaml_file(path)
    # if user requested a nested key, walk the dict using dot-splitting
    if key_part:
        parts = key_part.split(".")
        cur = data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                raise KeyError(
                    f"file resolver: key '{key_part}' not found in {path}",
                )
        data = cur
    data = OmegaConf.create(data)
    # Return as an OmegaConf node so structured content is preserved
    return data


# Register resolver with OmegaConf (Hydra will pick up this plugin module automatically)
OmegaConf.register_new_resolver(
    "file",
    _file_resolver,
    replace=True,
    use_cache=True,
)


def _merge_resolver(*args):
    """Resolve and merge multiple config fragments into a single OmegaConf node."""
    merged = OmegaConf.create()
    for arg in args:
        # Resolve any interpolations
        obj = OmegaConf.to_container(OmegaConf.create(arg), resolve=True)
        merged = OmegaConf.merge(merged, obj)
    return OmegaConf.create(merged)


OmegaConf.register_new_resolver("merge", _merge_resolver, replace=True)


def _hash_conf(*values, _root_=None):
    """Resolver wrapper for :func:`deckard.utils.hash_conf_values`."""
    return hash_conf_values(*values, _root_=_root_)


OmegaConf.register_new_resolver(
    "hash",
    _hash_conf,
    replace=True,
    use_cache=False,
)


def _coerce_artifact_path(path: Any) -> str:
    return str(path)


def _load_artifact_resolver(method_name: str, payload_kind: str):
    def _resolver(path: Any):
        artifact = ArtifactLoaderConfig(path=_coerce_artifact_path(path), payload_kind=payload_kind)
        return getattr(artifact, method_name)(_coerce_artifact_path(path))

    return _resolver


def _save_artifact_resolver(method_name: str, payload_kind: str):
    def _resolver(payload: Any, path: Any):
        artifact = ArtifactLoaderConfig(path=_coerce_artifact_path(path), payload_kind=payload_kind)
        getattr(artifact, method_name)(payload, _coerce_artifact_path(path))
        return _coerce_artifact_path(path)

    return _resolver


_ARTIFACT_LOADERS: dict[str, tuple[str, str]] = {
    "load_artifact": ("load", "data"),
    "load_data": ("load_data", "data"),
    "load_matrix": ("load_matrix", "data"),
    "load_vector": ("load_vector", "data"),
    "load_predictions": ("load_data", "data"),
    "load_benign_predictions": ("load_data", "data"),
    "load_adversarial_predictions": ("load_data", "data"),
    "load_attack_samples": ("load_data", "data"),
    "load_attack_predictions": ("load_data", "data"),
    "load_defended_predictions": ("load_data", "data"),
    "load_detected_predictions": ("load_data", "data"),
    "load_filtered_outputs": ("load_data", "data"),
    "load_probabilities": ("load_data", "data"),
    "load_scores": ("load_scores", "scores"),
    "load_model": ("load_model", "model"),
    "load_detector": ("load_model", "model"),
    "load_object": ("load_object", "object"),
}

_ARTIFACT_SAVERS: dict[str, tuple[str, str]] = {
    "save_artifact": ("save", "data"),
    "save_data": ("save_data", "data"),
    "save_matrix": ("save_data", "data"),
    "save_vector": ("save_data", "data"),
    "save_predictions": ("save_data", "data"),
    "save_benign_predictions": ("save_data", "data"),
    "save_adversarial_predictions": ("save_data", "data"),
    "save_attack_samples": ("save_data", "data"),
    "save_attack_predictions": ("save_data", "data"),
    "save_defended_predictions": ("save_data", "data"),
    "save_detected_predictions": ("save_data", "data"),
    "save_filtered_outputs": ("save_data", "data"),
    "save_probabilities": ("save_data", "data"),
    "save_scores": ("save_scores", "scores"),
    "save_model": ("save_object", "model"),
    "save_detector": ("save_object", "model"),
    "save_object": ("save_object", "object"),
}

for resolver_name, (method_name, payload_kind) in _ARTIFACT_LOADERS.items():
    OmegaConf.register_new_resolver(
        resolver_name,
        _load_artifact_resolver(method_name, payload_kind),
        replace=True,
        use_cache=False,
    )

for resolver_name, (method_name, payload_kind) in _ARTIFACT_SAVERS.items():
    OmegaConf.register_new_resolver(
        resolver_name,
        _save_artifact_resolver(method_name, payload_kind),
        replace=True,
        use_cache=False,
    )


logger = logging.getLogger(__name__)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "AttackConfig",
    "DetectorConfig",
    "ExperimentConfig",
    "DefenseConfig",
    "FileConfig",
    "ScorerDictConfig",
]

if SurvivalExperimentConfig is not None:
    __all__.append("SurvivalExperimentConfig")

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
        "error": {
            # Use RotatingFileHandler for log rotation
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(Path.cwd(), "error.log"),
            "formatter": "std",
            "level": logging.INFO,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB log file size limit
            "backupCount": 5,  # Keep up to 5 backup files
            "mode": "a",
        },
        "stream": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "deckard": {"handlers": ["default"], "propagate": True},
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
# Suppress third-party warnings from sklearn and ART internals.
warnings.filterwarnings("ignore", module=r"^sklearn(\.|$)")
warnings.filterwarnings("ignore", module=r"^art(\.|$)")
np.seterr(divide="ignore", invalid="ignore")
