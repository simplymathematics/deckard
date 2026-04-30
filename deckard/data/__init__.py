import logging

from .data import DataConfig, DataPipelineConfig

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