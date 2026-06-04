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
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from ._optional import OPTIONAL_RUNTIME_CLASS_PATHS
from . import config_resolvers as _config_resolvers
from .warnings_policy import apply_warning_policy

_load_yaml_file = _config_resolvers._load_yaml_file
_file_resolver = _config_resolvers._file_resolver
_merge_resolver = _config_resolvers._merge_resolver
register_core_resolvers = _config_resolvers.register_core_resolvers

apply_warning_policy()

_OPTIONAL_RUNTIME_CLASS_PATHS: dict[str, str] = dict(OPTIONAL_RUNTIME_CLASS_PATHS)


def _resolve_optional_runtime_class(
    class_path: str,
    *,
    enabled: bool,
) -> Any | None:
    if not enabled:
        return None
    try:
        from .utils import resolve_class

        return resolve_class(class_path)
    except Exception:  # pragma: no cover
        return None


from .data import DataConfig  # noqa E402
from .model import ModelConfig  # noqa E402
from .model.defense.base import DefenseConfig  # noqa E402

try:
    from .attack import AttackConfig  # noqa E402
except Exception:  # pragma: no cover
    # Keep top-level package importable for scopes that do not require attacks
    # when optional ART/Torch stacks fail during interpreter instrumentation.
    AttackConfig = None
from .artifacts import ArtifactLoaderMixin  # noqa E402
from .detector import DetectorConfig  # noqa E402

try:
    from .experiment import ExperimentConfig  # noqa E402
except Exception:  # pragma: no cover
    ExperimentConfig = None

try:
    from .experiment import SurvivalExperimentConfig  # noqa E402
except Exception:  # pragma: no cover
    SurvivalExperimentConfig = None
try:
    from .file import FileConfig  # noqa E402
except Exception:  # pragma: no cover
    FileConfig = None
from .score import ScorerDictConfig  # noqa E402
from .utils import hash_conf_values  # noqa E402

# from .plot import YellowbrickConfigList, YellowbrickPlotConfig


DECKARD_CONFIG_DIR = os.environ.get("DECKARD_CONFIG_DIR", "config")
DECKARD_DEFAULT_CONFIG_FILE = os.environ.get(
    "DECKARD_DEFAULT_CONFIG_FILE",
    "default.yaml",
)


register_core_resolvers()


def _hash_conf(*values, _root_=None):
    """Resolver wrapper for :func:`deckard.utils.hash_conf_values`."""
    return hash_conf_values(*values, _root_=_root_)


def _stage_params(stage: Any = "???", _root_=None):
    """Build stage-scoped params payload from canonical experiment components."""
    from .experiment.canon import (
        CANONICAL_EXPERIMENT_STAGE_COMPONENTS,
        normalize_experiment_stage,
    )

    root = _root_ if _root_ is not None else OmegaConf.create({})
    if stage in {None, "", "???"}:
        stage_token = OmegaConf.select(root, "stage", default="all")
    else:
        stage_token = stage
    try:
        stage = normalize_experiment_stage(stage_token)
    except Exception:
        stage = str(stage_token or "all")

    selected_components = CANONICAL_EXPERIMENT_STAGE_COMPONENTS.get(stage)
    if selected_components is None:
        component_union: set[str] = set()
        for components in CANONICAL_EXPERIMENT_STAGE_COMPONENTS.values():
            component_union.update(components)
        selected_components = tuple(sorted(component_union))

    component_path_overrides = {
        "sampler": "data.sampler",
        "pipeline": "data.pipeline",
        "trainer": "model.trainer",
        "framework": "library",
        "plugins": "plugins",
        "plot": "plot",
        "files": "files",
    }

    components: dict[str, Any] = {}
    for component in selected_components:
        path = component_path_overrides.get(component, component)
        value = OmegaConf.select(root, path, default=None)
        if value is not None:
            components[component] = value

    runtime_keys = (
        "library",
        "classifier",
        "evaluation_mode",
        "score_mode",
        "random_state",
        "optimizers",
        "directions",
        "report_trial_attrs",
        "pruning_enabled",
        "dvclive_enabled",
        "dvclive_dir",
    )
    runtime = {
        key: OmegaConf.select(root, key, default=None)
        for key in runtime_keys
        if OmegaConf.select(root, key, default=None) is not None
    }

    return {
        "stage": stage,
        "components": components,
        "runtime": runtime,
    }


OmegaConf.register_new_resolver(
    "hash",
    _hash_conf,
    replace=True,
    use_cache=False,
)

OmegaConf.register_new_resolver(
    "stage_params",
    _stage_params,
    replace=True,
    use_cache=False,
)

OmegaConf.register_new_resolver(
    "stage_hash_payload",
    _stage_params,
    replace=True,
    use_cache=False,
)


def _coerce_artifact_path(path: Any) -> str:
    return str(path)


def _load_artifact_resolver(method_name: str, payload_kind: str):
    def _resolver(path: Any):
        artifact = ArtifactLoaderMixin(
            path=_coerce_artifact_path(path),
            payload_kind=payload_kind,
        )
        return getattr(artifact, method_name)(_coerce_artifact_path(path))

    return _resolver


def _save_artifact_resolver(method_name: str, payload_kind: str):
    def _resolver(payload: Any, path: Any):
        artifact = ArtifactLoaderMixin(
            path=_coerce_artifact_path(path),
            payload_kind=payload_kind,
        )
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

np.seterr(divide="ignore", invalid="ignore")
