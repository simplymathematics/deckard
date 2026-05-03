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

# Import declarations to register example configs with ConfigStore
from . import declarations  # noqa: F401

logger = logging.getLogger(__name__)

try:
    from .fairness import FairlearnDataConfig

    _ = FairlearnDataConfig
except ImportError:  # pragma: no cover
    logger.debug("Fairlearn not found. FairlearnDataConfig is unavailable.")

try:
    from .anjana import AnjanaDataConfig

    _ = AnjanaDataConfig
except ImportError:  # pragma: no cover
    logger.debug("Anjana not found. AnjanaDataConfig is unavailable.")

try:
    from .pytorch import PytorchDataConfig, PytorchCustomDataConfig

    _ = (PytorchDataConfig, PytorchCustomDataConfig)
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

if "FairlearnDataConfig" in globals():
    __all__.append("FairlearnDataConfig")
if "AnjanaDataConfig" in globals():
    __all__.append("AnjanaDataConfig")
if "PytorchDataConfig" in globals():
    __all__.extend(["PytorchDataConfig", "PytorchCustomDataConfig"])
