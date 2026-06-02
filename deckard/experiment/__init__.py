"""Public experiment configuration exports.

The :mod:`deckard.experiment` package exposes the default experiment
orchestration config and an optional survival-specific extension.
"""

import logging

from .._optional import load_optional_export, load_optional_surface_exports
from .base import ExperimentConfig
from .canon import (
    CANONICAL_EXPERIMENT_PIPELINE_STAGES,
    normalize_experiment_pipeline_stage,
)
from .dvc import (
    DVCExperimentMixin,
    build_dvc_stage_name,
)
from .repro import (
    DVCReproPlugin,
    build_dvc_cmd,
    build_dvc_stage_plan,
    extract_dvc_file_aliases,
    generate_dvc_pipeline,
)
from .power import (
    DVCPowerMixin,
    DVCPowerPlugin,
    build_power_hook_bundle,
    build_power_plugin_hooks,
)

logger = logging.getLogger(__name__)

_OPTIONAL_EXPERIMENT_SURFACE = "deckard.experiment"

__all__ = [
    "ExperimentConfig",
    "CANONICAL_EXPERIMENT_PIPELINE_STAGES",
    "build_dvc_cmd",
    "build_dvc_stage_name",
    "build_dvc_stage_plan",
    "extract_dvc_file_aliases",
    "generate_dvc_pipeline",
    "DVCExperimentMixin",
    "DVCReproPlugin",
    "normalize_experiment_pipeline_stage",
    "DVCPowerMixin",
    "DVCPowerPlugin",
    "build_power_hook_bundle",
    "build_power_plugin_hooks",
]


def _load_torch_experiment_symbols() -> bool:
    return bool(
        load_optional_surface_exports(
            _OPTIONAL_EXPERIMENT_SURFACE,
            module_globals=globals(),
            exported_names=__all__,
            family="pytorch",
        ),
    )


def _load_lifelines_experiment_symbols() -> bool:
    return bool(
        load_optional_surface_exports(
            _OPTIONAL_EXPERIMENT_SURFACE,
            module_globals=globals(),
            exported_names=__all__,
            family="lifelines",
        ),
    )


def __getattr__(name: str):
    value = load_optional_export(
        _OPTIONAL_EXPERIMENT_SURFACE,
        name,
        module_globals=globals(),
        exported_names=__all__,
    )
    if value is not None:
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
