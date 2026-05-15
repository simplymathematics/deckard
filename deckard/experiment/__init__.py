"""Public experiment configuration exports.

The :mod:`deckard.experiment` package exposes the default experiment
orchestration config and an optional survival-specific extension.
"""

import logging

from .base import ExperimentConfig

logger = logging.getLogger(__name__)

try:
    from ..frameworks.pytorch.experiment import TorchExperimentConfig

    _ = TorchExperimentConfig
except Exception:  # pragma: no cover
    logger.debug(
        "PyTorch not found. TorchExperimentConfig is unavailable.",
    )

try:
    from ..plugins.lifelines.experiment import SurvivalExperimentConfig

    _ = SurvivalExperimentConfig
except Exception:  # pragma: no cover
    logger.debug(
        "Lifelines not found. SurvivalExperimentConfig is unavailable.",
    )


__all__ = ["ExperimentConfig"]

if "TorchExperimentConfig" in globals():
    __all__.append("TorchExperimentConfig")

if "SurvivalExperimentConfig" in globals():
    __all__.append("SurvivalExperimentConfig")
