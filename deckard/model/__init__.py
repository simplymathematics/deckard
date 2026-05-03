"""Public model and defense configuration exports.

The :mod:`deckard.model` package exposes the standard model pipeline together
with optional fairness-aware and PyTorch-backed variants when those optional
dependencies are installed.
"""

import logging
from typing import Any
from .base import ModelConfig
from .defend import DefenseConfig, DefensePipelineConfig

# Import declarations to register example configs with ConfigStore
from . import declarations  # noqa: F401

ScorerDictConfig = Any

logger = logging.getLogger(__name__)

try:
    from .survival import SurvivalModelConfig
except ImportError:  # pragma: no cover
    logger.debug("Lifelines not found. Survival model configs are unavailable.")

try:
    from .fairness import (
        FairlearnDefenseConfig,
        FairlearnModelConfig,
    )
except ImportError:  # pragma: no cover
    logger.debug(
        "Fairlearn not found. Fairlearn model configs are unavailable.",
    )

try:
    from .anjana import AnjanaModelConfig
except ImportError:  # pragma: no cover
    logger.debug("Anjana not found. Anjana model configs are unavailable.")

try:
    import torch
    from .pytorch import PytorchModelConfig
except ImportError:
    logger.debug("Torch not found. Cannot use torch features.")


__all__ = ["ModelConfig", "DefenseConfig", "DefensePipelineConfig"]

if "FairlearnDefenseConfig" in globals():
    __all__.extend(
        [
            "FairlearnDefenseConfig",
            "FairlearnModelConfig",
        ],
    )
if "AnjanaModelConfig" in globals():
    __all__.extend(["AnjanaModelConfig"])
if "PytorchModelConfig" in globals():
    __all__.extend(["PytorchModelConfig"])
