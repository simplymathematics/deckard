"""Public experiment configuration exports.

The :mod:`deckard.experiment` package exposes the default experiment
orchestration config and an optional survival-specific extension.
"""

import logging

from .base import ExperimentConfig

logger = logging.getLogger(__name__)

__all__ = ["ExperimentConfig"]


def _load_torch_experiment_symbols() -> bool:
    try:
        from ..frameworks.pytorch.experiment import TorchExperimentConfig
    except Exception:  # pragma: no cover
        logger.debug("PyTorch not found. TorchExperimentConfig is unavailable.")
        return False

    globals()["TorchExperimentConfig"] = TorchExperimentConfig
    if "TorchExperimentConfig" not in __all__:
        __all__.append("TorchExperimentConfig")
    return True


def _load_lifelines_experiment_symbols() -> bool:
    try:
        from ..plugins.lifelines.experiment import SurvivalExperimentConfig
    except Exception:  # pragma: no cover
        logger.debug("Lifelines not found. SurvivalExperimentConfig is unavailable.")
        return False

    globals()["SurvivalExperimentConfig"] = SurvivalExperimentConfig
    if "SurvivalExperimentConfig" not in __all__:
        __all__.append("SurvivalExperimentConfig")
    return True


def __getattr__(name: str):
    if name == "TorchExperimentConfig" and _load_torch_experiment_symbols():
        return globals()[name]
    if name == "SurvivalExperimentConfig" and _load_lifelines_experiment_symbols():
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
