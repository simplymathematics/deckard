import base  # noqa: F401
import layers  # noqa: F401
import tempfile, logging, os, warnings
from sklearn.exceptions import UndefinedMetricWarning

# Semantic Version
__version__ = "0.30"

# pylint: disable=C0103

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.FileHandler",
            "filename": os.path.join(tempfile.gettempdir(), "deckard.log"),
            "mode": "a",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.DEBUG,
        },
    },
    "loggers": {
        "deckard": {"handlers": ["default"]},
        "tests": {"handlers": ["test"], "level": "DEBUG", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
