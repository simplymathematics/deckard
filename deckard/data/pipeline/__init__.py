"""Data pipeline package exports.

This package centralizes pipeline-oriented config classes while maintaining
backward compatibility with DataPipelineConfig in data.base.
"""

import logging

from ..base import DataPipelineConfig
from .core import (
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


__all__ = [
    "DataPipeline",
    "DataPipelineConfig",
    "DefaultDataPipelineConfig",
    "AnjanaDataPipelineConfig",
    "FairlearnDataPipelineConfig",
]

if "PytorchDataPipelineConfig" in globals():
    __all__.append("PytorchDataPipelineConfig")
