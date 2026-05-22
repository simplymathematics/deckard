"""Canonical data runtime stage, mode, timing, and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Final, Mapping, TypedDict

from ..plugins import HookPlugin
from ..utils import (
    coerce_to_list,
    instantiate_plugin_spec,
    load_class,
    normalize_plugin_specs,
)


CANONICAL_DATA_METHODS: Final[tuple[str, ...]] = (
    "load",
    "sample",
    "pipeline",
)

CANONICAL_DATA_STAGES: Final[tuple[str, ...]] = (
    "pre-load",
    "pre-sample",
    "post-sample",
    "post-pipeline",
    "all",
    "auto",
)

CANONICAL_DATA_SCORE_MODES: Final[tuple[str, ...]] = (
    "train",
    "test",
    "val",
    "all",
)

CANONICAL_DATA_TIMES: Final[tuple[str, ...]] = (
    "data_load_time",
    "data_sample_time",
    "data_pipeline_time",
    "data_score_time",
)

CANONICAL_DATA_RUNTIME_FIELDS: Final[tuple[str, ...]] = (
    "score_dict",
    "files",
    "times",
    "_X",
    "_y",
    "train_indices",
    "test_indices",
    "val_indices",
    "X_train",
    "y_train",
    "X_test",
    "y_test",
    "X_val",
    "y_val",
    "train_n",
    "test_n",
    "val_n",
)

DEFAULT_DATA_SCORE_STAGE: Final[str] = "post-pipeline"
DEFAULT_DATA_SCORE_MODE: Final[str] = "test"


class BaseFiles(TypedDict, total=False):
    params_file: str | None


class DataFiles(TypedDict, total=False):
    """Canonical data persistence aliases used by DataConfig.__call__."""

    data_file: str | None
    post_sample_data_file: str | None
    post_pipeline_data_file: str | None
    params_file: str | None
    score_file: str | None
    train_labels_file: str | None
    test_labels_file: str | None
    val_labels_file: str | None
    metadata_file: str | None


class DataTimes(TypedDict, total=False):
    """Canonical data runtime timing keys (plus optional extensions)."""

    data_load_time: float | None
    data_sample_time: float | None
    data_pipeline_time: float | None
    data_score_time: float | None


_STAGE_ALIASES: Final[dict[str, str]] = {
    "pre-load": "pre-load",
    "preload": "pre-load",
    "before-load": "pre-load",
    "before_load": "pre-load",
    "pre-sample": "pre-sample",
    "pre_sample": "pre-sample",
    "presample": "pre-sample",
    "post-sample": "post-sample",
    "post_sample": "post-sample",
    "postsample": "post-sample",
    "post-pipeline": "post-pipeline",
    "post_pipeline": "post-pipeline",
    "postpipeline": "post-pipeline",
}

_MODE_ALIASES: Final[dict[str, str]] = {
    "train": "train",
    "training": "train",
    "test": "test",
    "eval": "test",
    "evaluation": "test",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "all": "all",
}

_EVENT_ALIASES: Final[dict[str, str]] = {
    "pre": "before",
    "before": "before",
    "post": "after",
    "after": "after",
}


def ensure_canonical_times(times: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return an extensible timing dict with canonical keys present."""
    merged = {key: None for key in CANONICAL_DATA_TIMES}
    if times:
        merged.update(dict(times))
    return merged


def merge_data_files(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> DataFiles:
    """Merge two file alias mappings into one canonical DataFiles payload."""
    merged: dict[str, Any] = {}
    if base:
        merged.update(dict(base))
    if override:
        merged.update(dict(override))
    return merged  # type: ignore[return-value]


def merge_files(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> DataFiles:
    """Backward-compatible alias for merge_data_files."""
    return merge_data_files(base, override)


def ensure_data_runtime_contract(target: Any) -> Any:
    """Populate canonical runtime attributes on a DataConfig-like object."""
    target.files = merge_data_files(getattr(target, "files", None), None)
    target.times = ensure_canonical_times(getattr(target, "times", None))
    if not hasattr(target, "score_dict") or getattr(target, "score_dict") is None:
        target.score_dict = {}

    for field in CANONICAL_DATA_RUNTIME_FIELDS:
        if field in {"score_dict", "files", "times"}:
            continue
        if not hasattr(target, field):
            setattr(target, field, None)
    return target


def stage_hook_token(stage: str) -> str:
    """Convert canonical stage names into hook-safe tokens."""
    key = str(stage).strip().lower().replace(" ", "-")
    if key in _STAGE_ALIASES:
        return _STAGE_ALIASES[key].replace("-", "_")
    raise ValueError(
        f"Unknown data hook stage '{stage}'. Must be one of {list(CANONICAL_DATA_STAGES)}",
    )


def normalize_data_score_mode(mode: str) -> str:
    """Normalize score split mode names to canonical tokens."""
    key = str(mode).strip().lower().replace(" ", "-")
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    raise ValueError(
        f"Unknown data score mode '{mode}'. Must be one of {list(CANONICAL_DATA_SCORE_MODES)}",
    )


def normalize_data_score_stage(value: str) -> str:
    """Normalize score stage aliases to canonical split tokens."""
    return normalize_data_score_mode(value)


@dataclass(eq=False, kw_only=True)
class DataPluginRuntimeMixin:
    """Reusable plugin orchestration and runtime-state copy behavior."""

    def _copy_runtime_state_to(self, target: Any) -> None:
        runtime_fields = [
            "score_dict",
            "data_load_time",
            "data_sample_time",
            "_X",
            "_y",
            "train_indices",
            "test_indices",
            "val_indices",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "X_val",
            "y_val",
            "train_n",
            "test_n",
            "val_n",
            "pipeline_fit_n",
            "pipeline_transform_n",
            "pipeline_fit_time",
            "pipeline_transform_time",
            "pipeline_y_fit_n",
            "pipeline_y_fit_time",
            "pipeline_y_transform_n",
            "pipeline_y_transform_time",
        ]
        for attr in runtime_fields:
            if hasattr(self, attr):
                setattr(target, attr, getattr(self, attr, None))

    def _instantiate_plugin(self, plugin_spec: Any):
        return instantiate_plugin_spec(plugin_spec, loader=load_class)

    def _get_plugins(self) -> list:
        if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
            plugin_specs = normalize_plugin_specs(getattr(self, "plugins", []))
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs: Any) -> list[Any]:
        hook_outputs: list[Any] = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs


@dataclass(eq=False, kw_only=True)
class ScoringOrchestratorMixin(DataPluginRuntimeMixin):
    """Stage-driven score orchestration mixin for data runtimes."""

    default_stage: Final[str] = DEFAULT_DATA_SCORE_STAGE
    stage_aliases: ClassVar[dict[str, str]] = _STAGE_ALIASES
    mode_aliases: ClassVar[dict[str, str]] = _MODE_ALIASES
    _score_orchestration_active: bool = True

    def _iter_configured_score_stages(self) -> list[str]:
        scorer = getattr(self, "scorer", None)
        configured = getattr(scorer, "configured_scorers", None)
        if not isinstance(configured, dict) or len(configured) == 0:
            return [self.default_stage]

        raw_stages: list[str] = []
        for scorer_cfg in configured.values():
            stage_value = getattr(scorer_cfg, "stage", None)
            if stage_value in [None, "", []]:
                raw_stages.append(self.default_stage)
                continue
            if isinstance(stage_value, str):
                raw_stages.append(stage_value)
                continue
            for token in coerce_to_list(stage_value):
                raw_stages.append(str(token))

        if len(raw_stages) == 0:
            return [self.default_stage]
        return raw_stages

    def _expand_canonical_score_stages(self, raw_stages: list[str]) -> list[str]:
        canonical = list(CANONICAL_DATA_STAGES)
        ordered = [stage for stage in canonical if stage not in {"all", "auto"}]
        expanded: list[str] = []

        for token in raw_stages:
            normalized = str(token).strip().lower().replace("_", "-")
            if normalized in {"", "auto"}:
                expanded.append(self.default_stage)
                continue
            if normalized == "all":
                expanded.extend(ordered)
                continue
            if normalized in ordered:
                expanded.append(normalized)
                continue
            raise ValueError(
                f"Unsupported data score stage '{token}'. "
                f"Expected one of {ordered + ['all', 'auto']}",
            )

        deduped: list[str] = []
        for stage in ordered:
            if stage in expanded and stage not in deduped:
                deduped.append(stage)
        return deduped or [self.default_stage]

    def _configure_score_orchestration_plugins(self) -> None:
        stage_to_hook = {
            "pre-load": "before_load_data",
            "pre-sample": "before_sample",
            "post-sample": "after_sample",
            "post-pipeline": "after_pipeline",
        }
        stage_tokens = self._expand_canonical_score_stages(
            self._iter_configured_score_stages(),
        )
        score_plugins = [
            HookPlugin(
                hook_name=stage_to_hook[stage],
                method_name="_score_orchestration_hook",
                method_kwargs={"stage": stage},
            )
            for stage in stage_tokens
            if stage in stage_to_hook
        ]
        if len(score_plugins) == 0:
            return
        if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
            self._plugin_objects = []
        self._plugin_objects.extend(score_plugins)

    def _score_orchestration_hook(self, stage: str, **kwargs: Any):
        if not self._score_orchestration_active:
            return None
        mode = kwargs.pop("mode", None)
        mode = normalize_data_score_mode(mode or getattr(self, "score_split", "test"))
        score_kwargs = kwargs.pop("score_kwargs", None) or {}
        self._run_score_stage_hooks("before", stage, score_kwargs=score_kwargs)
        score_fn = getattr(self, "score", None)
        if not callable(score_fn):
            raise AttributeError(f"{type(self).__name__} has no callable 'score' method")
        result = score_fn(mode=mode, stage=stage, **score_kwargs)
        plugin_scores = self._run_score_stage_hooks("after", stage, scores=result)
        if isinstance(result, dict):
            for plugin_score in plugin_scores:
                if isinstance(plugin_score, dict):
                    result.update(plugin_score)
            if self.score_dict is None:
                self.score_dict = {}
            for key, value in result.items():
                if (
                    key in self.score_dict
                    and isinstance(self.score_dict.get(key), dict)
                    and isinstance(value, dict)
                ):
                    self.score_dict[key].update(value)
                else:
                    self.score_dict[key] = value
        return result

    def _run_score_stage_hooks(
        self,
        when: str,
        stage: str,
        **kwargs: Any,
    ) -> list[Any]:
        event = str(when).strip().lower()
        if event not in {"before", "after", "pre", "post"}:
            raise ValueError(f"Score hook event must be 'before' or 'after', got {when}")
        event = _EVENT_ALIASES[event]
        stage_token = stage_hook_token(stage)
        stage = stage_token.replace("_", "-")
        stage_kwargs = {"stage": stage, **kwargs}
        outputs: list[Any] = []
        outputs.extend(
            self._run_plugin_hook(
                f"{event}_score_{stage_token}",
                **stage_kwargs,
            ),
        )
        outputs.extend(self._run_plugin_hook(f"{event}_score", **stage_kwargs))
        return outputs


__all__ = [
    "CANONICAL_DATA_METHODS",
    "CANONICAL_DATA_STAGES",
    "CANONICAL_DATA_SCORE_MODES",
    "CANONICAL_DATA_TIMES",
    "CANONICAL_DATA_RUNTIME_FIELDS",
    "DEFAULT_DATA_SCORE_STAGE",
    "DEFAULT_DATA_SCORE_MODE",
    "DataFiles",
    "DataTimes",
    "ensure_canonical_times",
    "ensure_data_runtime_contract",
    "stage_hook_token",
    "normalize_data_score_mode",
    "normalize_data_score_stage",
    "merge_data_files",
    "merge_files",
    "DataPluginRuntimeMixin",
    "ScoringOrchestratorMixin",
]
