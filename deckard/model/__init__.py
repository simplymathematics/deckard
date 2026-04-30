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
    from .pytorch import PytorchCustomPretrainedModelConfig, PytorchModelConfig
except ImportError:
    logger.debug("Torch not found. Cannot use torch features.")