"""Public data configuration exports.

The :mod:`deckard.data` package provides the default tabular data pipeline along
with optional fairness-aware and PyTorch-specific data configuration classes.
Optional exports are only available when their dependencies are installed.
"""

import logging

from .._optional import (
    get_optional_surface_export_names,
    load_optional_export,
    load_optional_surface_exports,
)
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

_OPTIONAL_DATA_SURFACE = "deckard.data"
_OPTIONAL_DATA_EXPORTS = get_optional_surface_export_names(_OPTIONAL_DATA_SURFACE)


def _clear_optional_exports(*names: str) -> None:
    """Drop stale optional exports before conditional re-import on reload."""
    module_globals = globals()
    for name in names:
        module_globals.pop(name, None)


_clear_optional_exports(
    *_OPTIONAL_DATA_EXPORTS,
)

load_optional_surface_exports(
    _OPTIONAL_DATA_SURFACE,
    module_globals=globals(),
)


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


def __getattr__(name: str):
    value = load_optional_export(
        _OPTIONAL_DATA_SURFACE,
        name,
        module_globals=globals(),
    )
    if value is not None:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
