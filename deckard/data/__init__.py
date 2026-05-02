"""Public data configuration exports.

The :mod:`deckard.data` package provides the default tabular data pipeline along
with optional fairness-aware and PyTorch-specific data configuration classes.
Optional exports are only available when their dependencies are installed.
"""

import logging

from .base import DataConfig, DataPipelineConfig
from .sample import (
    BaseSampler,
    SplitSampler,
    KFoldSampler,
    ShuffleSampler,
    register_sampler_configs,
)

logger = logging.getLogger(__name__)

try:
    from .fairness import FairnessDataConfig
except ImportError:  # pragma: no cover
    logger.debug("Fairlearn not found. FairnessDataConfig is unavailable.")

try:
    import torch
    from .pytorch import PytorchDataConfig, PytorchCustomDataConfig
except ImportError:
    logger.debug("Torch not found.")


__all__ = [
    "DataConfig",
    "DataPipelineConfig",
    "BaseSampler",
    "SplitSampler",
    "KFoldSampler",
    "ShuffleSampler",
    "register_sampler_configs",
]

if "FairnessDataConfig" in globals():
    __all__.append("FairnessDataConfig")
if "PytorchDataConfig" in globals():
    __all__.extend(["PytorchDataConfig", "PytorchCustomDataConfig"])
