"""Data pipeline package exports.

This package centralizes pipeline-oriented config classes while maintaining
backward compatibility with DataPipelineConfig in data.base.
"""

import logging

from .base import (
    AnjanaDataPipelineConfig,
    DataPipeline,
    DefaultDataPipelineConfig,
    FairlearnDataPipelineConfig,
)

logger = logging.getLogger(__name__)

try:
    from ...frameworks.pytorch.data import PytorchDataPipelineConfig

    _ = PytorchDataPipelineConfig
except Exception:  # pragma: no cover
    logger.debug("Torch not found. PytorchDataPipelineConfig is unavailable.")

try:
    from ..base import DataPipelineConfig
except Exception:  # pragma: no cover
    DataPipelineConfig = None


__all__ = [
    "DataPipeline",
    "DefaultDataPipelineConfig",
    "AnjanaDataPipelineConfig",
    "FairlearnDataPipelineConfig",
]

if DataPipelineConfig is not None:
    __all__.append("DataPipelineConfig")

if "PytorchDataPipelineConfig" in globals():
    __all__.append("PytorchDataPipelineConfig")
