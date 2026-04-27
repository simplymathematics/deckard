import logging

from .data import DataConfig, DataPipelineConfig
from .fairness import FairnessDataConfig

logger = logging.getLogger(__name__)
try:
    import torch
    from .pytorch import PytorchDataConfig, PytorchCustomDataConfig
except ImportError:
    logger.warning("Torch not found.")