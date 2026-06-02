"""Public model and defense configuration exports.

The :mod:`deckard.model` package exposes the standard model pipeline together
with optional fairness-aware and PyTorch-backed variants when those optional
dependencies are installed.
"""

import logging
from typing import Any

from .._optional import (
    get_optional_surface_export_names,
    load_optional_export,
    load_optional_surface_exports,
)
from .base import ModelConfig
from .defense.base import DefenseConfig
from .defense.detector import DetectorDefenseConfig
from .defense.postprocessor import PostprocessorDefenseConfig
from .defense.preprocessor import PreprocessorDefenseConfig
from .defense.regularizer import RegularizerDefenseConfig
from .defense.trainer import TrainerDefenseConfig
from .defense.transformer import TransformerDefenseConfig
from .trainers import (
    BaseTrainer,
    PartialFitPruningTrainer,
    PartialFitTrainer,
    PretrainedTrainer,
    PruningTrainer,
    PytorchTrainer,
    SklearnTrainer,
)

ScorerDictConfig = Any

logger = logging.getLogger(__name__)

_OPTIONAL_MODEL_SURFACE = "deckard.model"
_OPTIONAL_MODEL_EXPORTS = get_optional_surface_export_names(_OPTIONAL_MODEL_SURFACE)


def _clear_optional_exports(*names: str) -> None:
    """Drop stale optional exports before conditional re-import on reload."""
    module_globals = globals()
    for name in names:
        module_globals.pop(name, None)


_clear_optional_exports(
    *_OPTIONAL_MODEL_EXPORTS,
)


def _load_optional_model_symbols() -> None:
    """Best-effort eager load of optional model symbols.

    This avoids hard failures at import time while still populating globals
    when optional dependencies are available.
    """
    load_optional_surface_exports(
        _OPTIONAL_MODEL_SURFACE,
        module_globals=globals(),
        exported_names=__all__,
    )


def _load_fairlearn_model_symbols() -> bool:
    """Best-effort loader for optional fairlearn model configs.

    Returns:
        True when symbols were loaded into module globals, else False.
    """
    return bool(
        load_optional_surface_exports(
            _OPTIONAL_MODEL_SURFACE,
            module_globals=globals(),
            exported_names=__all__,
            family="fairlearn",
        ),
    )


def _load_lifelines_model_symbols() -> bool:
    return bool(
        load_optional_surface_exports(
            _OPTIONAL_MODEL_SURFACE,
            module_globals=globals(),
            exported_names=__all__,
            family="lifelines",
        ),
    )


def _load_anjana_model_symbols() -> bool:
    return bool(
        load_optional_surface_exports(
            _OPTIONAL_MODEL_SURFACE,
            module_globals=globals(),
            exported_names=__all__,
            family="anjana",
        ),
    )


def _load_torch_model_symbols() -> bool:
    return bool(
        load_optional_surface_exports(
            _OPTIONAL_MODEL_SURFACE,
            module_globals=globals(),
            exported_names=__all__,
            family="pytorch",
        ),
    )


__all__ = [
    "ModelConfig",
    "DefenseConfig",
    "DetectorDefenseConfig",
    "PreprocessorDefenseConfig",
    "PostprocessorDefenseConfig",
    "TrainerDefenseConfig",
    "RegularizerDefenseConfig",
    "TransformerDefenseConfig",
    "BaseTrainer",
    "SklearnTrainer",
    "PretrainedTrainer",
    "PartialFitTrainer",
    "PartialFitPruningTrainer",
    "PruningTrainer",
    "PytorchTrainer",
]


def __getattr__(name: str):
    """Lazily resolve optional model symbols on first attribute access."""
    value = load_optional_export(
        _OPTIONAL_MODEL_SURFACE,
        name,
        module_globals=globals(),
        exported_names=__all__,
    )
    if value is not None:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
