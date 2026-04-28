import logging
from .base import ModelConfig
from .defend import DefenseConfig
from .fairness import FairnessModelConfig

logger = logging.getLogger(__name__)

try:
    import torch
    from .pytorch import PytorchCustomPretrainedModelConfig, PytorchModelConfig
except ImportError:
    logger.warning("Torch not found. Cannot use torch features.")