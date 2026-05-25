"""Public data configuration exports.

The :mod:`deckard.data` package provides the default tabular data pipeline along
with optional fairness-aware and PyTorch-specific data configuration classes.
Optional exports are only available when their dependencies are installed.
"""

import logging

from ..plugins import is_plugin_available
from .base import DataConfig
from .declarations import DatasetDeclaration, discover_dataset_declarations
from .pipeline import DataConfig as PipelineDataConfig
from .pipeline import DataPipeline
from .sample import (
    BaseSampler,
    KFoldSampler,
    ShuffleSampler,
    SplitSampler,
    register_sampler_configs,
)

logger = logging.getLogger(__name__)


def _clear_optional_exports(*names: str) -> None:
    """Drop stale optional exports before conditional re-import on reload."""
    module_globals = globals()
    for name in names:
        module_globals.pop(name, None)


_clear_optional_exports(
    "PytorchDataConfig",
    "PytorchCustomDataConfig",
    "FairlearnDataConfig",
    "AnjanaDataConfig",
)

try:
    from .pipeline import PytorchDataConfig

    _ = PytorchDataConfig
except Exception:  # pragma: no cover
    logger.debug("Torch not found. PytorchDataConfig is unavailable.")


try:
    from ..frameworks.pytorch.data import PytorchCustomDataConfig, PytorchDataConfig

    _ = (PytorchDataConfig, PytorchCustomDataConfig)
except Exception:
    logger.debug("Torch not found.")

if is_plugin_available("fairlearn"):
    try:
        from ..plugins.fairlearn.data import FairlearnDataConfig

        _ = FairlearnDataConfig
    except Exception:
        logger.debug("Fairlearn plugin import failed.")
else:
    logger.debug("Fairlearn plugin dependencies not installed.")

if is_plugin_available("anjana"):
    try:
        from ..plugins.anjana.data import AnjanaDataConfig

        _ = AnjanaDataConfig
    except Exception:
        logger.debug("Anjana plugin import failed.")
else:
    logger.debug("Anjana plugin dependencies not installed.")


__all__ = [
    "DataConfig",
    "DataPipeline",
    "PipelineDataConfig",
    "BaseSampler",
    "SplitSampler",
    "KFoldSampler",
    "ShuffleSampler",
    "register_sampler_configs",
    "DatasetDeclaration",
    "discover_dataset_declarations",
]

if "PytorchDataConfig" in globals():
    __all__.extend(["PytorchDataConfig", "PytorchCustomDataConfig"])

if "FairlearnDataConfig" in globals():
    __all__.append("FairlearnDataConfig")

if "AnjanaDataConfig" in globals():
    __all__.append("AnjanaDataConfig")
