"""Data pipeline package exports.

This package centralizes pipeline-oriented runtime helpers with DataConfig as
the canonical data config surface.
"""

import logging

from .base import DataConfig, DataPipeline

logger = logging.getLogger(__name__)

try:
    from ...frameworks.pytorch.data import PytorchDataConfig

    _ = PytorchDataConfig
except Exception:  # pragma: no cover
    logger.debug("Torch not found. PytorchDataConfig is unavailable.")


__all__ = [
    "DataPipeline",
    "DataConfig",
]

if "PytorchDataConfig" in globals():
    __all__.append("PytorchDataConfig")
