"""Canonical experiment runtime contract helpers.

This module defines the implementation-level ExperimentConfig contract used by
the runtime orchestration layer and provides normalization helpers for stage and
score-mode routing.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Mapping, TypedDict

from ..attack.canon import ATTACK_RUNTIME_STAGE_ALIASES
from ..data.canon import CANONICAL_DATA_STAGES
from ..detector.canon import DETECTOR_RUNTIME_STAGE_ALIASES
from ..model.canon import CANONICAL_MODEL_DEFENSE_STAGES
from ..plugins import HookPlugin
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
    manifest: dict[str, Any] = {
        "experiment_name": getattr(target, "experiment_name", None),
        "library": getattr(target, "library", None),
        "classifier": getattr(target, "classifier", None),
        "evaluation_mode": getattr(target, "evaluation_mode", None),
        "score_mode": getattr(target, "score_mode", None),
        "random_state": getattr(target, "random_state", None),
    }

    for component_name in ("data", "model", "defense", "attack", "detector", "score"):
        component = getattr(target, component_name, None)
        if component is None:
            manifest[component_name] = None
            continue
        manifest[component_name] = {
            "type": f"{component.__class__.__module__}.{component.__class__.__name__}",
            "alias": getattr(component, "alias", None),
        }

    if runtime_kwargs:
        manifest["runtime_kwargs"] = dict(runtime_kwargs)
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
    payload: dict[str, Any] = {
        "params": dict(params_manifest),
        "stage": normalize_experiment_stage(stage),
        "component": str(component).strip().lower(),
        "identity": dict(identity or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
