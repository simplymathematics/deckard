"""Public model and defense configuration exports.

The :mod:`deckard.model` package exposes the standard model pipeline together
with optional fairness-aware and PyTorch-backed variants when those optional
dependencies are installed.
"""

import logging
from .base import ModelConfig
from .defend import DefenseConfig

logger = logging.getLogger(__name__)

try:
    from .fairness import FairnessDefenseConfig, FairnessModelConfig
except ImportError:  # pragma: no cover
    logger.debug("Fairlearn not found. Fairness model configs are unavailable.")

try:
    import torch
    from .pytorch import PytorchModelConfig
except ImportError:
    logger.debug("Torch not found. Cannot use torch features.")


__all__ = ["ModelConfig", "DefenseConfig"]

if "FairnessDefenseConfig" in globals():
    __all__.extend(["FairnessDefenseConfig", "FairnessModelConfig"])
if "PytorchModelConfig" in globals():
    __all__.extend(["PytorchModelConfig"])
