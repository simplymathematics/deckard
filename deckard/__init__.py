import logging
from .data import *
from .model import *
from .attack import *
from .data import DataConfig, data_parser, data_main
from .model import ModelConfig, model_parser, model_main
from .attack import AttackConfig, attack_parser, attack_main

# Import configs, parsers, and mains from each module

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

__all__ = [
    "DataConfig",
    "data_parser",
    "data_main",
    "ModelConfig",
    "model_parser",
    "model_main",
    "AttackConfig",
    "attack_parser",
    "attack_main",
]
