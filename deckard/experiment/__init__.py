"""Public experiment configuration exports.

The :mod:`deckard.experiment` package exposes the default experiment
orchestration config and an optional survival-specific extension.
"""

import logging

from .base import ExperimentConfig
from .canon import (
    CANONICAL_EXPERIMENT_PIPELINE_STAGES,
    normalize_experiment_pipeline_stage,
)
from .dvc import (
    build_dvc_cmd,
    build_dvc_stage_name,
    build_dvc_stage_plan,
    extract_dvc_file_aliases,
    generate_dvc_pipeline,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExperimentConfig",
    "CANONICAL_EXPERIMENT_PIPELINE_STAGES",
    "build_dvc_cmd",
    "build_dvc_stage_name",
    "build_dvc_stage_plan",
    "extract_dvc_file_aliases",
    "generate_dvc_pipeline",
    "normalize_experiment_pipeline_stage",
]


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
