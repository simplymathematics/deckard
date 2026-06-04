"""Declarative optional-family registry and lazy export helpers.

This module centralizes metadata for optional framework/plugin families so
package surfaces can resolve availability, exported symbols, and runtime class
paths from one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from . import declarations as declarations_mod


@dataclass(frozen=True)
class OptionalExport:
    """Declarative description of one optional symbol exposed on a surface."""

    surface: str
    name: str
    module: str
    attr: str | None = None
    runtime_key: str | None = None

    @property
    def attribute_name(self) -> str:
        return self.attr or self.name

    @property
    def class_path(self) -> str:
        return f"{self.module}.{self.attribute_name}"


@dataclass(frozen=True)
class OptionalFamily:
    """Metadata for one optional framework or plugin family."""

    kind: str
    module: str
    required_imports: tuple[str, ...]
    exports: tuple[OptionalExport, ...] = ()


OPTIONAL_FAMILY_REGISTRY: dict[str, OptionalFamily] = {
    "anjana": OptionalFamily(
        kind="plugin",
        module="deckard.plugins.anjana",
        required_imports=("anjana", "pycanon"),
        exports=(
            OptionalExport(
                surface="deckard.data",
                name="AnjanaDataConfig",
                module="deckard.plugins.anjana.data",
                runtime_key="anjana_data_config",
            ),
            OptionalExport(
                surface="deckard.model",
                name="AnjanaModelConfig",
                module="deckard.plugins.anjana.model",
            ),
            OptionalExport(
                surface="deckard.score",
                name="DefaultAnjanaScorerDictConfig",
                module="deckard.plugins.anjana.score",
            ),
            OptionalExport(
                surface="deckard.score",
                name="DefaultAnjanaDataScorerDictConfig",
                module="deckard.plugins.anjana.score",
            ),
            OptionalExport(
                surface="deckard.score",
                name="DefaultAnjanaModelScorerDictConfig",
                module="deckard.plugins.anjana.score",
            ),
        ),
    ),
    "fairlearn": OptionalFamily(
        kind="plugin",
        module="deckard.plugins.fairlearn",
        required_imports=("fairlearn",),
        exports=(
            OptionalExport(
                surface="deckard.data",
                name="FairlearnDataConfig",
                module="deckard.plugins.fairlearn.data",
                runtime_key="fairlearn_data_config",
            ),
            OptionalExport(
                surface="deckard.model",
                name="FairlearnDefenseConfig",
                module="deckard.plugins.fairlearn.model",
            ),
            OptionalExport(
                surface="deckard.model",
                name="FairlearnModelConfig",
                module="deckard.plugins.fairlearn.model",
                runtime_key="fairlearn_model_config",
            ),
            OptionalExport(
                surface="deckard.model",
                name="FairlearnPytorchModelConfig",
                module="deckard.plugins.fairlearn.model",
                runtime_key="fairlearn_pytorch_model_config",
            ),
            OptionalExport(
                surface="deckard.score",
                name="DefaultFairlearnClassificationScorerDictConfig",
                module="deckard.plugins.fairlearn.score",
            ),
            OptionalExport(
                surface="deckard.score",
                name="DefaultFairlearnDataScorerDictConfig",
                module="deckard.plugins.fairlearn.score",
            ),
            OptionalExport(
                surface="deckard.score",
                name="DefaultFairlearnRegressionScorerDictConfig",
                module="deckard.plugins.fairlearn.score",
            ),
            OptionalExport(
                surface="deckard.score",
                name="DefaultFairlearnScorerDictConfig",
                module="deckard.plugins.fairlearn.score",
            ),
            OptionalExport(
                surface="deckard.score",
                name="FairlearnScorerDictConfig",
                module="deckard.plugins.fairlearn.score",
                runtime_key="fairlearn_scorer_config",
            ),
        ),
    ),
    "lifelines": OptionalFamily(
        kind="plugin",
        module="deckard.plugins.lifelines",
        required_imports=("lifelines",),
        exports=(
            OptionalExport(
                surface="deckard.experiment",
                name="SurvivalExperimentConfig",
                module="deckard.plugins.lifelines.experiment",
            ),
            OptionalExport(
                surface="deckard.model",
                name="SurvivalModelConfig",
                module="deckard.plugins.lifelines.model",
            ),
            OptionalExport(
                surface="deckard.plot",
                name="SurvivalSeabornPlotConfigList",
                module="deckard.plugins.lifelines.plot",
            ),
            OptionalExport(
                surface="deckard.plot",
                name="SurvivalSeabornPlotterConfig",
                module="deckard.plugins.lifelines.plot",
            ),
            OptionalExport(
                surface="deckard.score",
                name="DefaultLifelinesConfig",
                module="deckard.plugins.lifelines.score",
            ),
        ),
    ),
    "openattack": OptionalFamily(
        kind="plugin",
        module="deckard.plugins.openattack",
        required_imports=("OpenAttack",),
    ),
    "pytorch": OptionalFamily(
        kind="framework",
        module="deckard.frameworks.pytorch",
        required_imports=("torch",),
        exports=(
            OptionalExport(
                surface="deckard.data",
                name="PytorchCustomDataConfig",
                module="deckard.frameworks.pytorch.data",
            ),
            OptionalExport(
                surface="deckard.data",
                name="PytorchDataConfig",
                module="deckard.frameworks.pytorch.data",
            ),
            OptionalExport(
                surface="deckard.experiment",
                name="TorchExperimentConfig",
                module="deckard.frameworks.pytorch.experiment",
            ),
            OptionalExport(
                surface="deckard.model",
                name="PytorchModelConfig",
                module="deckard.frameworks.pytorch.model",
                runtime_key="pytorch_model_config",
            ),
        ),
    ),
    "seaborn": OptionalFamily(
        kind="plugin",
        module="deckard.plugins.seaborn",
        required_imports=("seaborn",),
        exports=(
            OptionalExport(
                surface="deckard.plot",
                name="SeabornPlotConfig",
                module="deckard.plugins.seaborn.plot",
            ),
            OptionalExport(
                surface="deckard.plot",
                name="SeabornPlotConfigList",
                module="deckard.plugins.seaborn.plot",
            ),
        ),
    ),
    "sklearn": OptionalFamily(
        kind="framework",
        module="sklearn",
        required_imports=("sklearn",),
    ),
    "textattack": OptionalFamily(
        kind="plugin",
        module="deckard.plugins.textattack",
        required_imports=("textattack",),
    ),
    "datasets": OptionalFamily(
        kind="plugin",
        module="deckard.plugins.datasets",
        required_imports=("datasets",),
    ),
    "transformers_framework": OptionalFamily(
        kind="framework",
        module="deckard.frameworks.transformers",
        required_imports=("transformers",),
    ),
    "yellowbrick": OptionalFamily(
        kind="plugin",
        module="deckard.plugins.yellowbrick",
        required_imports=("yellowbrick",),
        exports=(
            OptionalExport(
                surface="deckard.plot",
                name="YellowbrickConfigList",
                module="deckard.plugins.yellowbrick.plot",
            ),
            OptionalExport(
                surface="deckard.plot",
                name="YellowbrickPlotConfig",
                module="deckard.plugins.yellowbrick.plot",
            ),
        ),
    ),
}


def get_optional_family_names(*, kind: str | None = None) -> tuple[str, ...]:
    names = tuple(OPTIONAL_FAMILY_REGISTRY)
    if kind is None:
        return names
    return tuple(name for name in names if OPTIONAL_FAMILY_REGISTRY[name].kind == kind)


def get_optional_family(name: str, *, kind: str | None = None) -> OptionalFamily:
    family = OPTIONAL_FAMILY_REGISTRY[name]
    if kind is not None and family.kind != kind:
        raise KeyError(f"Optional family '{name}' is not a {kind} family")
    return family


def get_optional_family_required_imports(name: str) -> tuple[str, ...]:
    return get_optional_family(name).required_imports


def is_optional_family_available(name: str, *, kind: str | None = None) -> bool:
    family = get_optional_family(name, kind=kind)
    return all(
        declarations_mod.is_package_available(package_name)
        for package_name in family.required_imports
    )


def get_optional_family_module(name: str, *, kind: str | None = None):
    family = get_optional_family(name, kind=kind)
    if not is_optional_family_available(name, kind=kind):
        required = ", ".join(family.required_imports or (name,))
        raise ImportError(
            f"Optional family '{name}' is not available. Install optional dependencies "
            f"for it ({required}).",
        )
    return import_module(family.module)


def iter_optional_exports(
    surface: str,
    *,
    family: str | None = None,
) -> tuple[tuple[str, OptionalExport], ...]:
    items: list[tuple[str, OptionalExport]] = []
    family_names = (family,) if family is not None else get_optional_family_names()
    for family_name in family_names:
        for export in OPTIONAL_FAMILY_REGISTRY[family_name].exports:
            if export.surface == surface:
                items.append((family_name, export))
    return tuple(items)


def get_optional_surface_export_names(
    surface: str,
    *,
    family: str | None = None,
) -> tuple[str, ...]:
    return tuple(
        export.name for _, export in iter_optional_exports(surface, family=family)
    )


def get_optional_export(
    surface: str,
    name: str,
) -> tuple[str, OptionalExport] | None:
    for family_name, export in iter_optional_exports(surface):
        if export.name == name:
            return family_name, export
    return None


def load_optional_export(
    surface: str,
    name: str,
    *,
    module_globals: dict[str, Any],
    exported_names: list[str] | None = None,
) -> Any | None:
    export_spec = get_optional_export(surface, name)
    if export_spec is None:
        return None

    family_name, export = export_spec
    if not is_optional_family_available(family_name):
        return None

    try:
        module = import_module(export.module)
        value = getattr(module, export.attribute_name)
    except Exception:  # pragma: no cover
        return None

    module_globals[name] = value
    if exported_names is not None and name not in exported_names:
        exported_names.append(name)
    return value


def load_optional_surface_exports(
    surface: str,
    *,
    module_globals: dict[str, Any],
    exported_names: list[str] | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for _, export in iter_optional_exports(surface, family=family):
        value = load_optional_export(
            surface,
            export.name,
            module_globals=module_globals,
            exported_names=exported_names,
        )
        if value is not None:
            loaded[export.name] = value
    return loaded


OPTIONAL_RUNTIME_CLASS_PATHS: dict[str, str] = {
    export.runtime_key: export.class_path
    for _, family in OPTIONAL_FAMILY_REGISTRY.items()
    for export in family.exports
    if export.runtime_key is not None
}
