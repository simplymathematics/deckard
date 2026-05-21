"""Public data configuration exports.

The :mod:`deckard.data` package provides the default tabular data pipeline along
with optional fairness-aware and PyTorch-specific data configuration classes.
Optional exports are only available when their dependencies are installed.
"""

import logging

from .base import DataConfig, DataPipelineConfig
from .pipeline import (
    AnjanaDataPipelineConfig,
    DefaultDataPipelineConfig,
    FairlearnDataPipelineConfig,
)
from .sample import (
    BaseSampler,
    KFoldSampler,
    ShuffleSampler,
    SplitSampler,
    register_sampler_configs,
)

logger = logging.getLogger(__name__)

try:
    from .pipeline import PytorchDataPipelineConfig

    _ = PytorchDataPipelineConfig
except Exception:  # pragma: no cover
    logger.debug("Torch not found. PytorchDataPipelineConfig is unavailable.")


try:
    from ..frameworks.pytorch.data import PytorchCustomDataConfig, PytorchDataConfig

    _ = (PytorchDataConfig, PytorchCustomDataConfig)
except Exception:
    logger.debug("Torch not found.")

try:
    from ..plugins.fairlearn.data import FairlearnDataConfig

    _ = FairlearnDataConfig
except Exception:
    logger.debug("Fairlearn not found.")

try:
    from ..plugins.anjana.data import AnjanaDataConfig

    _ = AnjanaDataConfig
except Exception:
    logger.debug("Anjana not found.")


__all__ = [
    "DataConfig",
    "DataPipelineConfig",
    "DataPipelineMixin",
    "DefaultDataPipelineConfig",
    "AnjanaDataPipelineConfig",
    "FairlearnDataPipelineConfig",
    "BaseSampler",
    "SplitSampler",
    "KFoldSampler",
    "ShuffleSampler",
    "register_sampler_configs",
]

if "PytorchDataPipelineConfig" in globals():
    __all__.append("PytorchDataPipelineConfig")


if "PytorchDataConfig" in globals():
    __all__.extend(["PytorchDataConfig", "PytorchCustomDataConfig"])

if "FairlearnDataConfig" in globals():
    __all__.append("FairlearnDataConfig")

if "AnjanaDataConfig" in globals():
    __all__.append("AnjanaDataConfig")
