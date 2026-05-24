"""Public model and defense configuration exports.

The :mod:`deckard.model` package exposes the standard model pipeline together
with optional fairness-aware and PyTorch-backed variants when those optional
dependencies are installed.
"""

import logging
from typing import Any

from .base import ModelConfig
from .defense.base import DefenseConfig, DefensePipelineConfig
from .defense import (
    DefaultDefenseConfig,
    DefaultPytorchDefenseConfig,
    DefaultSklearnDefenseConfig,
)
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


def _load_optional_model_symbols() -> None:
    """Best-effort eager load of optional model symbols.

    This avoids hard failures at import time while still populating globals
    when optional dependencies are available.
    """
    _load_fairlearn_model_symbols()
    _load_lifelines_model_symbols()
    _load_anjana_model_symbols()
    _load_torch_model_symbols()


def _load_fairlearn_model_symbols() -> bool:
    """Best-effort loader for optional fairlearn model configs.

    Returns:
        True when symbols were loaded into module globals, else False.
    """
    try:
        from ..plugins.fairlearn.model import (
            FairlearnDefenseConfig,
            FairlearnModelConfig,
            FairlearnPytorchModelConfig,
        )
    except ImportError:  # pragma: no cover
        return False

    globals()["FairlearnDefenseConfig"] = FairlearnDefenseConfig
    globals()["FairlearnModelConfig"] = FairlearnModelConfig
    globals()["FairlearnPytorchModelConfig"] = FairlearnPytorchModelConfig
    if "__all__" in globals():
        for symbol in (
            "FairlearnDefenseConfig",
            "FairlearnModelConfig",
            "FairlearnPytorchModelConfig",
        ):
            if symbol not in __all__:
                __all__.append(symbol)
    return True


def _load_lifelines_model_symbols() -> bool:
    try:
        from ..plugins.lifelines.model import SurvivalModelConfig
    except ImportError:  # pragma: no cover
        return False

    globals()["SurvivalModelConfig"] = SurvivalModelConfig
    if "__all__" in globals() and "SurvivalModelConfig" not in __all__:
        __all__.append("SurvivalModelConfig")
    return True


def _load_anjana_model_symbols() -> bool:
    try:
        from ..plugins.anjana.model import AnjanaModelConfig
    except ImportError:  # pragma: no cover
        return False

    globals()["AnjanaModelConfig"] = AnjanaModelConfig
    if "__all__" in globals() and "AnjanaModelConfig" not in __all__:
        __all__.append("AnjanaModelConfig")
    return True


def _load_torch_model_symbols() -> bool:
    try:
        from ..frameworks.pytorch.model import PytorchModelConfig
    except ImportError:  # pragma: no cover
        return False

    globals()["PytorchModelConfig"] = PytorchModelConfig
    if "__all__" in globals() and "PytorchModelConfig" not in __all__:
        __all__.append("PytorchModelConfig")
    return True


__all__ = [
    "ModelConfig",
    "DefaultDefenseConfig",
    "DefaultSklearnDefenseConfig",
    "DefaultPytorchDefenseConfig",
    "DefenseConfig",
    "DefensePipelineConfig",
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

if "FairlearnDefenseConfig" in globals():
    __all__.extend(
        [
            "FairlearnDefenseConfig",
            "FairlearnModelConfig",
            "FairlearnPytorchModelConfig",
        ],
    )
if "AnjanaModelConfig" in globals():
    __all__.extend(["AnjanaModelConfig"])
if "PytorchModelConfig" in globals():
    __all__.extend(["PytorchModelConfig"])
if "SurvivalModelConfig" in globals():
    __all__.extend(["SurvivalModelConfig"])


def __getattr__(name: str):
    """Lazily resolve optional model symbols on first attribute access."""
    fairlearn_symbols = {
        "FairlearnDefenseConfig",
        "FairlearnModelConfig",
        "FairlearnPytorchModelConfig",
    }
    lifelines_symbols = {"SurvivalModelConfig"}
    anjana_symbols = {"AnjanaModelConfig"}
    torch_symbols = {"PytorchModelConfig"}
    if name in fairlearn_symbols and _load_fairlearn_model_symbols():
        return globals()[name]
    if name in lifelines_symbols and _load_lifelines_model_symbols():
        return globals()[name]
    if name in anjana_symbols and _load_anjana_model_symbols():
        return globals()[name]
    if name in torch_symbols and _load_torch_model_symbols():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
