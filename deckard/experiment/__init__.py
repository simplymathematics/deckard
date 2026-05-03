"""Public experiment configuration exports.

The :mod:`deckard.experiment` package exposes the default experiment
orchestration config and an optional survival-specific extension.
"""

import logging

from .base import ExperimentConfig

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401
    from .torch_experiment import TorchExperimentConfig
except ImportError:  # pragma: no cover
    logger.debug(
        "PyTorch not found. TorchExperimentConfig is unavailable.",
    )

try:
    import lifelines  # noqa: F401
    from .survival import SurvivalExperimentConfig
except ImportError:  # pragma: no cover
    logger.debug(
        "Lifelines not found. SurvivalExperimentConfig is unavailable.",
    )


__all__ = ["ExperimentConfig"]

if "TorchExperimentConfig" in globals():
    __all__.append("TorchExperimentConfig")

if "SurvivalExperimentConfig" in globals():
    __all__.append("SurvivalExperimentConfig")
