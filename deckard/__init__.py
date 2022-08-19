from deckard import base, layers
import logging
# Semantic Version
__version__ = "0.20"

# pylint: disable=C0103

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "std": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M",
        }
    },
    "handlers": {
        "default": {
            "class": "logging.NullHandler",
        },
        "test": {
            "class": "logging.StreamHandler",
            "formatter": "std",
            "level": logging.INFO,
        },
    },
    "loggers": {
        "art": {"handlers": ["default"]},
        "tests": {"handlers": ["test"], "level": "INFO", "propagate": True},
    },
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)