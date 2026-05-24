"""Canonical experiment runtime contract helpers.

This module defines the implementation-level ExperimentConfig contract used by
the runtime orchestration layer and provides normalization helpers for stage and
score-mode routing.
"""

from __future__ import annotations

import hashlib
import json
import pkgutil
from typing import Any, Final, Mapping, TypedDict

from ..attack.canon import ATTACK_RUNTIME_STAGE_ALIASES, AttackFiles
from ..data.canon import CANONICAL_DATA_STAGES, DataFiles
from ..detector.canon import DETECTOR_RUNTIME_STAGE_ALIASES, DetectorFiles
from ..frameworks import __path__ as FRAMEWORKS_PACKAGE_PATHS
from ..orchestration import CANONICAL_RUNTIME_METHODS
from ..model.canon import (
    CANONICAL_MODEL_DEFENSE_STAGES,
    CANONICAL_MODEL_TRAINER_ALIASES,
    DefenseFiles,
    ModelFiles,
)
from ..plot.canon import CANON_PLOT_BACKENDS
from ..plugins import HookPlugin
from ..plugins import __all__ as PLUGIN_NAMESPACE_EXPORTS
from ..plugins.base import HookBundle


CANONICAL_EXPERIMENT_SCORE_MODES: Final[tuple[str, ...]] = (
    "pre-sample",
    "train",
    "test",
    "val",
)

CANONICAL_EXPERIMENT_STAGES: Final[tuple[str, ...]] = (
    "load",
    "sample",
    "train",
    "defense",
    "attack",
    "score",
    "persist",
    "all",
)

# Expanded stage order used by DVC/base orchestration for component-level hooks.
CANONICAL_EXPERIMENT_PIPELINE_STAGES: Final[tuple[str, ...]] = (
    "load",
    "sample",
    "pipeline",
    "data_score",
    "data_persist",
    "apply_fit_defense",
    "train",
    "apply_predict_defense",
    "model_score",
    "model_persist",
    "generation",
    "attack_score",
    "attack_persist",
    "detector-train",
    "detector-defense",
    "detector_score",
    "detector_persist",
    "score",
    "persist",
)

def _build_stage_component_mapping() -> dict[str, tuple[str, ...]]:
    stage_sets: dict[str, set[str]] = {
        "load": set(),
        "sample": set(),
        "train": set(),
        "defense": set(),
        "attack": set(),
        "score": set(),
        "persist": set(),
    }

    plugin_families = {
        token
        for token in PLUGIN_NAMESPACE_EXPORTS
        if token.islower() and token not in {"get_plugin"}
    }
    framework_families = {
        module.name
        for module in pkgutil.iter_modules(FRAMEWORKS_PACKAGE_PATHS)
        if module.ispkg
    }

    data_methods = set(CANONICAL_RUNTIME_METHODS)
    if "load" in data_methods:
        stage_sets["load"].add("data")
    if "sample" in data_methods:
        stage_sets["sample"].update({"data", "sampler"})
    if "pipeline" in data_methods:
        stage_sets["sample"].add("pipeline")
        stage_sets["train"].add("pipeline")
        stage_sets["defense"].add("pipeline")

    data_stages = set(CANONICAL_DATA_STAGES)
    if "pre-load" in data_stages:
        stage_sets["load"].add("data")
    if {"pre-sample", "post-sample"} & data_stages:
        stage_sets["sample"].update({"data", "sampler"})
    if "post-pipeline" in data_stages:
        stage_sets["sample"].add("pipeline")
        stage_sets["train"].add("data")
        stage_sets["score"].add("data")

    if len(CANONICAL_MODEL_TRAINER_ALIASES) > 0:
        stage_sets["train"].add("trainer")
        stage_sets["defense"].add("trainer")
        stage_sets["attack"].add("trainer")
    if len(CANONICAL_MODEL_DEFENSE_STAGES) > 0:
        stage_sets["train"].add("model")
        stage_sets["defense"].update({"model", "defense"})
        stage_sets["attack"].update({"model", "defense"})
        stage_sets["score"].update({"model", "defense"})

    attack_stages = set(ATTACK_RUNTIME_STAGE_ALIASES.values())
    if {"pre-attack", "post-attack"} & attack_stages:
        stage_sets["attack"].add("attack")
        stage_sets["score"].add("attack")

    detector_stages = set(DETECTOR_RUNTIME_STAGE_ALIASES.values())
    if {"pre-fit", "post-fit", "pre-detect", "post-detect"} & detector_stages:
        stage_sets["defense"].add("detector")
        stage_sets["attack"].add("detector")
        stage_sets["score"].add("detector")

    if len(plugin_families) > 0:
        for stage in ("load", "sample", "train", "defense", "attack", "score"):
            stage_sets[stage].add("plugins")

    if len(framework_families) > 0:
        for stage in ("sample", "train", "defense", "attack", "score"):
            stage_sets[stage].add("framework")

    if len(CANON_PLOT_BACKENDS) > 0:
        stage_sets["score"].add("plot")

    stage_sets["score"].add("score")

    persist_components = {"experiment", "files"}
    for stage in ("load", "sample", "train", "defense", "attack", "score"):
        persist_components.update(stage_sets[stage])
    stage_sets["persist"].update(persist_components)

    component_order = (
        "data",
        "sampler",
        "pipeline",
        "framework",
        "plugins",
        "model",
        "trainer",
        "defense",
        "detector",
        "attack",
        "score",
        "plot",
        "experiment",
        "files",
    )
    stage_primary_preferences: dict[str, tuple[str, ...]] = {
        "load": ("data",),
        "sample": ("data",),
        "train": ("model", "trainer"),
        "defense": ("detector", "defense", "model"),
        "attack": ("attack", "model"),
        "score": ("score", "plot"),
        "persist": ("experiment",),
    }

    mapping: dict[str, tuple[str, ...]] = {}
    for stage, components in stage_sets.items():
        ordered = [component for component in component_order if component in components]
        if len(ordered) == 0:
            continue
        preferred = stage_primary_preferences.get(stage, ())
        for preferred_component in reversed(preferred):
            if preferred_component in ordered:
                ordered.remove(preferred_component)
                ordered.insert(0, preferred_component)
        mapping[stage] = tuple(ordered)
    return mapping


CANONICAL_EXPERIMENT_STAGE_COMPONENTS: Final[dict[str, tuple[str, ...]]] = (
    _build_stage_component_mapping()
)

CANONICAL_EXPERIMENT_STAGE_PRIMARY_COMPONENTS: Final[dict[str, str]] = {
    stage: components[0]
    for stage, components in CANONICAL_EXPERIMENT_STAGE_COMPONENTS.items()
}

CANONICAL_EXPERIMENT_RUN_MODE_ALIASES: Final[dict[str, str]] = {
    "single": "single",
    "run": "single",
    "runmode.run": "single",
    "multirun": "multirun",
    "multi": "multirun",
    "runmode.multirun": "multirun",
}

CANONICAL_EXPERIMENT_TIMES: Final[tuple[str, ...]] = (
    "experiment_total_time",
    "data_load_time",
    "data_sample_time",
    "model_training_time",
    "attack_time",
    "detector_time",
    "score_time",
)

CANONICAL_EXPERIMENT_RUNTIME_FIELDS: Final[tuple[str, ...]] = (
    "score_dict",
    "files",
    "times",
    "outputs",
    "params",
)

CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_VERSION: Final[str] = (
    "deckard.experiment.runtime.v1"
)
CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_PREFIX: Final[str] = (
    "deckard.experiment.runtime.v"
)

CANONICAL_EXPERIMENT_COMPONENT_STAGES: Final[dict[str, tuple[str, ...]]] = {
    "data": tuple(
        stage for stage in CANONICAL_DATA_STAGES if stage not in {"all", "auto"}
    ),
    "model": CANONICAL_MODEL_DEFENSE_STAGES,
    "attack": tuple(sorted(set(ATTACK_RUNTIME_STAGE_ALIASES.values()))),
    "detector": tuple(sorted(set(DETECTOR_RUNTIME_STAGE_ALIASES.values()))),
    "score": ("score",),
}

CANONICAL_EXPERIMENT_CACHE_STAGES: Final[tuple[str, ...]] = (
    "sample",
    "train",
    "defense",
    "attack",
    "score",
)


class ExperimentTimes(TypedDict, total=False):
    """Canonical experiment timing keys (plus optional extensions)."""

    experiment_total_time: float | None
    data_load_time: float | None
    data_sample_time: float | None
    model_training_time: float | None
    attack_time: float | None
    detector_time: float | None
    score_time: float | None


class ExperimentOutputs(TypedDict, total=False):
    """Canonical experiment output bucket for cached runtime payload metadata."""

    scores: dict[str, Any]
    files: dict[str, Any]
    cache: dict[str, Any]
    hooks: dict[str, Any]


class ExperimentFiles(TypedDict, total=False):
    """Canonical experiment-level persistence aliases."""

    params_file: str | None
    score_file: str | None
    log_file: str | None
    error_file: str | None


def _typed_dict_keys(*schemas: type[Any]) -> set[str]:
    keys: set[str] = set()
    for schema in schemas:
        keys.update(getattr(schema, "__annotations__", {}).keys())
    return keys


def _rank_output_key(key: str) -> tuple[int, str]:
    token = str(key).strip().lower()
    if token.endswith("model_file"):
        return (0, token)
    if token.endswith("data_file"):
        return (1, token)
    if "prediction" in token:
        return (2, token)
    if "probabilit" in token:
        return (3, token)
    if token.endswith("score_file"):
        return (4, token)
    if token.endswith("params_file"):
        return (5, token)
    if token.endswith("log_file") or token.endswith("error_file"):
        return (6, token)
    return (7, token)


def _select_output_keys(
    keys: set[str],
    *,
    include: tuple[str, ...] = (),
    include_any: tuple[str, ...] = (),
    include_suffix: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> tuple[str, ...]:
    selected: set[str] = set()
    for key in keys:
        token = str(key).strip().lower()
        if token in exclude:
            continue
        if token in include:
            selected.add(token)
            continue
        if include_any and any(fragment in token for fragment in include_any):
            selected.add(token)
            continue
        if include_suffix and any(token.endswith(suffix) for suffix in include_suffix):
            selected.add(token)
            continue
    return tuple(sorted(selected, key=_rank_output_key))


def build_experiment_stage_output_keys() -> dict[str, tuple[str, ...]]:
    """Build expanded pipeline stage output keys from canon file TypedDict schemas."""
    data_file_keys = _typed_dict_keys(DataFiles)
    model_file_keys = _typed_dict_keys(ModelFiles)
    defense_file_keys = _typed_dict_keys(DefenseFiles)
    attack_file_keys = _typed_dict_keys(AttackFiles)
    detector_file_keys = _typed_dict_keys(DetectorFiles)
    experiment_file_keys = _typed_dict_keys(ExperimentFiles)

    model_runtime_keys = _select_output_keys(
        model_file_keys,
        include_suffix=("model_file",),
        include_any=("prediction", "probabilit"),
        exclude=("score_file",),
    )
    detector_runtime_keys = _select_output_keys(
        detector_file_keys,
        include_suffix=("model_file",),
        include_any=("prediction", "probabilit"),
        exclude=("score_file",),
    )
    attack_runtime_keys = _select_output_keys(
        attack_file_keys,
        include_suffix=("attack_file",),
        include_any=("prediction",),
        exclude=("score_file",),
    )
    defense_runtime_keys = tuple(
        sorted(
            set(
                _select_output_keys(
                    defense_file_keys,
                    include_suffix=("model_file",),
                    include_any=("prediction", "probabilit"),
                    exclude=("score_file",),
                ),
            )
            | set(detector_runtime_keys),
            key=_rank_output_key,
        ),
    )
    data_runtime_keys = _select_output_keys(
        data_file_keys,
        include_suffix=("data_file",),
    )

    return {
        "load": data_runtime_keys,
        "sample": data_runtime_keys,
        "pipeline": data_runtime_keys,
        "data-persist": data_runtime_keys,
        "train": model_runtime_keys,
        "model-persist": model_runtime_keys,
        "detector-train": _select_output_keys(
            detector_file_keys,
            include_suffix=("model_file",),
        ),
        "defense": defense_runtime_keys,
        "detector-persist": detector_runtime_keys,
        "generation": attack_runtime_keys,
        "attack-persist": attack_runtime_keys,
        "score": _select_output_keys(
            experiment_file_keys | model_file_keys | attack_file_keys,
            include=("score_file",),
        ),
        "persist": _select_output_keys(
            experiment_file_keys,
            include=("params_file", "score_file", "log_file", "error_file"),
        ),
    }


CANONICAL_EXPERIMENT_STAGE_OUTPUT_KEYS: Final[dict[str, tuple[str, ...]]] = (
    build_experiment_stage_output_keys()
)


_MODE_ALIASES: Final[dict[str, str]] = {
    "pre-sample": "pre-sample",
    "presample": "pre-sample",
    "pre_sample": "pre-sample",
    "train": "train",
    "training": "train",
    "test": "test",
    "eval": "test",
    "evaluation": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
}

_STAGE_ALIASES: Final[dict[str, str]] = {
    "load": "load",
    "data-load": "load",
    "sample": "sample",
    "sampling": "sample",
    "pipeline": "sample",
    "train": "train",
    "training": "train",
    "defense": "defense",
    "attack": "attack",
    "score": "score",
    "scoring": "score",
    "persist": "persist",
    "persistence": "persist",
    "save": "persist",
    "all": "all",
}

_PIPELINE_STAGE_ALIASES: Final[dict[str, str]] = {
    "load": "load",
    "sample": "sample",
    "pipeline": "pipeline",
    "data-score": "data_score",
    "data_score": "data_score",
    "data-persist": "data_persist",
    "data_persist": "data_persist",
    "apply-fit-defense": "apply_fit_defense",
    "apply_fit_defense": "apply_fit_defense",
    "train": "train",
    "apply-predict-defense": "apply_predict_defense",
    "apply_predict_defense": "apply_predict_defense",
    "model-score": "model_score",
    "model_score": "model_score",
    "model-persist": "model_persist",
    "model_persist": "model_persist",
    "generation": "generation",
    "attack-score": "attack_score",
    "attack_score": "attack_score",
    "attack-persist": "attack_persist",
    "attack_persist": "attack_persist",
    "detector-train": "detector-train",
    "detector_train": "detector-train",
    "detector-defense": "detector-defense",
    "detector_defense": "detector-defense",
    "detector-score": "detector_score",
    "detector_score": "detector_score",
    "detector-persist": "detector_persist",
    "detector_persist": "detector_persist",
    "score": "score",
    "persist": "persist",
}


def ensure_canonical_experiment_times(
    times: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an extensible timing dict with canonical experiment keys present."""
    merged = {key: None for key in CANONICAL_EXPERIMENT_TIMES}
    if times:
        merged.update(dict(times))
    return merged


def normalize_experiment_score_mode(mode: str | None) -> str:
    """Normalize one experiment score mode alias to canonical token."""
    token = str(mode or "test").strip().lower().replace(" ", "-")
    resolved = _MODE_ALIASES.get(token)
    if resolved is None:
        raise ValueError(
            "Unknown experiment score mode "
            f"'{mode}'. Must be one of {list(CANONICAL_EXPERIMENT_SCORE_MODES)}",
        )
    return resolved


def normalize_experiment_score_modes(modes: Any) -> list[str]:
    """Normalize a scalar/list score-mode input into canonical mode list."""
    if modes is None:
        return ["test"]
    if isinstance(modes, (list, tuple)):
        raw_modes = list(modes)
    else:
        raw_modes = [modes]
    return [normalize_experiment_score_mode(mode) for mode in raw_modes]


def normalize_experiment_stage(stage: str | None) -> str:
    """Normalize one experiment stage alias to canonical token."""
    token = str(stage or "all").strip().lower().replace("_", "-")
    resolved = _STAGE_ALIASES.get(token)
    if resolved is None:
        raise ValueError(
            "Unknown experiment stage "
            f"'{stage}'. Must be one of {list(CANONICAL_EXPERIMENT_STAGES)}",
        )
    return resolved


def normalize_experiment_pipeline_stage(stage: str | None) -> str:
    """Normalize one expanded pipeline stage alias token."""
    token = str(stage or "persist").strip().lower().replace(" ", "-")
    token = token.replace("__", "_")
    resolved = _PIPELINE_STAGE_ALIASES.get(token)
    if resolved is None:
        raise ValueError(
            "Unknown experiment pipeline stage "
            f"'{stage}'. Must be one of {list(CANONICAL_EXPERIMENT_PIPELINE_STAGES)}",
        )
    return resolved


def ensure_experiment_runtime_contract(target: Any) -> Any:
    """Populate canonical runtime attributes on an ExperimentConfig-like object."""
    if not hasattr(target, "score_dict") or getattr(target, "score_dict") is None:
        target.score_dict = {}

    if not hasattr(target, "times") or getattr(target, "times") is None:
        target.times = {}
    target.times = ensure_canonical_experiment_times(getattr(target, "times", None))

    if not hasattr(target, "outputs") or getattr(target, "outputs") is None:
        target.outputs = {}
    if not hasattr(target, "params") or getattr(target, "params") is None:
        target.params = {}

    return target


def build_experiment_params_manifest(
    target: Any,
    *,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a lightweight, serializable params manifest for experiment runtime."""

    def _fingerprint_payload(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _component_manifest(component: Any) -> dict[str, Any]:
        base_payload: dict[str, Any] = {
            "type": f"{component.__class__.__module__}.{component.__class__.__name__}",
            "alias": getattr(component, "alias", None),
        }
        if hasattr(component, "to_dict") and callable(getattr(component, "to_dict")):
            try:
                payload = component.to_dict(for_hash=True)
                base_payload["fingerprint"] = _fingerprint_payload(payload)
            except Exception:
                base_payload["fingerprint"] = _fingerprint_payload(base_payload)
        else:
            base_payload["fingerprint"] = _fingerprint_payload(base_payload)
        return base_payload

    manifest: dict[str, Any] = {
        "experiment_name": getattr(target, "experiment_name", None),
        "library": getattr(target, "library", None),
        "classifier": getattr(target, "classifier", None),
        "evaluation_mode": getattr(target, "evaluation_mode", None),
        "score_mode": getattr(target, "score_mode", None),
        "random_state": getattr(target, "random_state", None),
    }
    # TODO: Correctly map components/sub components to existing *Config objects 
    for component_name in ("data", "model", "defense", "attack", "detector",  "score"):
        component = getattr(target, component_name, None)
        if component is None:
            manifest[component_name] = None
            continue
        manifest[component_name] = _component_manifest(component)

    attack_chain = getattr(target, "_attack_chain", None)
    if isinstance(attack_chain, (list, tuple)) and len(attack_chain) > 0:
        manifest["attack_chain"] = [
            _component_manifest(component)
            for component in attack_chain
            if component is not None
        ]

    if runtime_kwargs:
        manifest["runtime_kwargs"] = dict(runtime_kwargs)

    manifest["schema_version"] = CANONICAL_EXPERIMENT_RUNTIME_SCHEMA_VERSION
    return manifest


def _hook_stage_token(stage: str) -> str:
    return str(stage).strip().lower().replace(" ", "-").replace("-", "_")


def build_experiment_hook_graph() -> dict[str, list[dict[str, str]]]:
    """Build the canonical stage hook graph from component canon definitions."""
    graph: dict[str, list[dict[str, str]]] = {}
    for component, stages in CANONICAL_EXPERIMENT_COMPONENT_STAGES.items():
        component_nodes: list[dict[str, str]] = []
        for stage in stages:
            stage_token = _hook_stage_token(stage)
            component_nodes.append(
                {
                    "stage": stage,
                    "before": f"before_{stage_token}",
                    "after": f"after_{stage_token}",
                }
            )
        graph[component] = component_nodes
    return graph


def build_experiment_hook_plugins(
    method_name: str = "_experiment_stage_hook",
) -> list[HookPlugin]:
    """Construct HookPlugin objects from the canonical experiment hook graph."""
    plugins: list[HookPlugin] = []
    graph = build_experiment_hook_graph()
    for component, nodes in graph.items():
        for node in nodes:
            stage = node["stage"]
            for event, hook_name in (("before", node["before"]), ("after", node["after"])):
                plugins.append(
                    HookPlugin(
                        hook_name=hook_name,
                        method_name=method_name,
                        method_kwargs={
                            "component": component,
                            "stage": stage,
                            "event": event,
                        },
                    )
                )
    return plugins


def build_experiment_hook_bundle(
    name: str = "experiment-runtime",
    method_name: str = "_experiment_stage_hook",
) -> HookBundle:
    """Build a reusable HookBundle for ExperimentConfig stage orchestration."""
    return HookBundle(name=name, hooks=tuple(build_experiment_hook_plugins(method_name)))


def build_experiment_stage_cache_key(
    *,
    params_manifest: Mapping[str, Any],
    stage: str,
    component: str,
    identity: Mapping[str, Any] | None = None,
) -> str:
    """Build a deterministic stage cache key from params + stage identity."""
    stage_params = build_experiment_stage_params_subset(
        params_manifest=params_manifest,
        stage=stage,
        component=component,
    )
    payload: dict[str, Any] = {
        "params": stage_params,
        "stage": _normalize_stage_for_param_selection(stage),
        "component": str(component).strip().lower(),
        "identity": dict(identity or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_PIPELINE_STAGE_TO_RUNTIME_STAGE: Final[dict[str, str]] = {
    "load": "load",
    "sample": "sample",
    "pipeline": "sample",
    "data_score": "score",
    "data_persist": "persist",
    "apply_fit_defense": "defense",
    "train": "train",
    "apply_predict_defense": "defense",
    "model_score": "score",
    "model_persist": "persist",
    "generation": "attack",
    "attack_score": "score",
    "attack_persist": "persist",
    "detector-train": "defense",
    "detector-defense": "defense",
    "detector_score": "score",
    "detector_persist": "persist",
    "score": "score",
    "persist": "persist",
}

_MANIFEST_COMPONENT_KEY_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "data": ("data",),
    "sampler": ("data",),
    "pipeline": ("data",),
    "framework": (),
    "plugins": (),
    "model": ("model",),
    "trainer": ("model",),
    "defense": ("defense",),
    "detector": ("detector",),
    "attack": ("attack", "attack_chain"),
    "score": ("score",),
    "plot": (),
    "experiment": (),
    "files": (),
}

_MANIFEST_RUNTIME_KEY_PATHS: Final[tuple[str, ...]] = (
    "schema_version",
    "experiment_name",
    "library",
    "classifier",
    "evaluation_mode",
    "score_mode",
    "random_state",
)


def _normalize_stage_for_param_selection(stage: str | None) -> str:
    token = str(stage or "persist").strip().lower().replace(" ", "-")
    token = token.replace("__", "_")

    try:
        return normalize_experiment_pipeline_stage(token)
    except Exception:
        pass

    pipeline_tokens = list(_PIPELINE_STAGE_ALIASES.keys())
    for base in sorted(pipeline_tokens, key=len, reverse=True):
        if token == base or token.startswith(f"{base}-"):
            return normalize_experiment_pipeline_stage(base)

    return normalize_experiment_stage(token)


def _runtime_stage_for_param_selection(stage: str | None) -> str:
    normalized = _normalize_stage_for_param_selection(stage)
    return _PIPELINE_STAGE_TO_RUNTIME_STAGE.get(normalized, normalized)


def _manifest_component_keys_for_stage(
    *,
    stage: str | None,
    component: str | None,
) -> tuple[str, ...]:
    runtime_stage = _runtime_stage_for_param_selection(stage)
    stage_components = list(CANONICAL_EXPERIMENT_STAGE_COMPONENTS.get(runtime_stage, ()))

    component_token = str(component or "").strip().lower()
    if component_token != "":
        stage_components = [component_token]

    resolved: list[str] = []
    for stage_component in stage_components:
        for key in _MANIFEST_COMPONENT_KEY_ALIASES.get(stage_component, ()):  # noqa: B007
            if key not in resolved:
                resolved.append(key)
    return tuple(resolved)


def build_experiment_stage_param_key_paths(
    *,
    stage: str,
    component: str | None = None,
) -> tuple[str, ...]:
    """Build canonical params-manifest key paths relevant to one stage/component."""
    keys: list[str] = list(_MANIFEST_RUNTIME_KEY_PATHS)
    for key in _manifest_component_keys_for_stage(stage=stage, component=component):
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def _extract_mapping_by_key_paths(
    payload: Mapping[str, Any],
    key_paths: tuple[str, ...],
) -> dict[str, Any]:
    selected: dict[str, Any] = {}

    for key_path in key_paths:
        parts = [part for part in str(key_path).split(".") if part]
        if not parts:
            continue

        source: Any = payload
        found = True
        for part in parts:
            if not isinstance(source, Mapping) or part not in source:
                found = False
                break
            source = source[part]
        if not found:
            continue

        target: dict[str, Any] = selected
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = source

    return selected


def build_experiment_stage_params_subset(
    *,
    params_manifest: Mapping[str, Any],
    stage: str,
    component: str | None = None,
) -> dict[str, Any]:
    """Select stage-relevant params-manifest keys for DVC tracking and cache keys."""
    key_paths = build_experiment_stage_param_key_paths(stage=stage, component=component)
    return _extract_mapping_by_key_paths(params_manifest, key_paths)
