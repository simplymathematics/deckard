"""Public model and defense configuration exports.

The :mod:`deckard.model` package exposes the standard model pipeline together
with optional fairness-aware and PyTorch-backed variants when those optional
dependencies are installed.
"""

import logging
import sys
from typing import Any
from .base import ModelConfig
from .defense import (
    DefaultDefenseConfig,
    DefaultPytorchDefenseConfig,
    DefaultSklearnDefenseConfig,
)
from .defend import DefenseConfig, DefensePipelineConfig
from .detector import DetectorDefenseConfig
from .preprocessor import PreprocessorDefenseConfig
from .postprocessor import PostprocessorDefenseConfig
from .trainer import TrainerDefenseConfig
from .regularizer import RegularizerDefenseConfig
from .transformer import TransformerDefenseConfig

# Import declarations to register example configs with ConfigStore
from . import declarations  # noqa: F401

ScorerDictConfig = Any

logger = logging.getLogger(__name__)

try:
    from ..plugins.lifelines.model import SurvivalModelConfig

    _ = SurvivalModelConfig
except ImportError:  # pragma: no cover
    logger.debug("Lifelines not found. Survival model configs are unavailable.")

try:
    from ..plugins.fairlearn.model import (
        FairlearnDefenseConfig,
        FairlearnModelConfig,
        FairlearnPytorchModelConfig,
    )

    _ = (FairlearnDefenseConfig, FairlearnModelConfig, FairlearnPytorchModelConfig)
except ImportError:  # pragma: no cover
    logger.debug(
        "Fairlearn not found. Fairlearn model configs are unavailable.",
    )

try:
    from ..plugins.anjana.model import AnjanaModelConfig

    _ = AnjanaModelConfig
except ImportError:  # pragma: no cover
    logger.debug("Anjana not found. Anjana model configs are unavailable.")

try:
    from ..frameworks.pytorch.model import PytorchModelConfig

    _ = PytorchModelConfig
except ImportError:
    logger.debug("Torch not found. Cannot use torch features.")


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
