import logging
import warnings
from .data import DataConfig, data_parser, data_main
from .model import ModelConfig, model_parser, model_main
from .attack import AttackConfig, attack_parser, attack_main
from .utils import *
# Suppress UserWarnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress RuntimeWarnings from numpy
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Suppress ConvergenceWarnings from sklearn
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
