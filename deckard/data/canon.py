"""Canonical data runtime stage, mode, timing, and persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Final, Mapping, TypedDict

from ..plugins.base import OrchestratorBase, RuntimeBase


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
    "pipeline_fit_time",
    "pipeline_transform_time",
    "pipeline_y_fit_time",
    "pipeline_y_transform_time",
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
    pipeline_fit_time: float | None
    pipeline_transform_time: float | None
    pipeline_y_fit_time: float | None
    pipeline_y_transform_time: float | None


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

_RUNTIME_SPLIT_ALIASES: Final[dict[str, str]] = {
    "train": "train",
    "test": "test",
    "val": "val",
    "all": "all",
    "attack": "test",
    "attack-val": "val",
    "pre-sample": "all",
    "post-pipeline": "test",
    "post-sample": "test",
    "pre-load": "test",
    "auto": "test",
    "benign": "test",
    "adversarial": "test",
}

_SPLIT_DATA_ATTRS: Final[dict[str, tuple[str, str]]] = {
    "train": ("y_train", "X_train"),
    "test": ("y_test", "X_test"),
    "val": ("y_val", "X_val"),
    "all": ("_y", "_X"),
}

_SPLIT_SENSITIVE_ATTRS: Final[dict[str, str]] = {
    "train": "_sensitive_train",
    "test": "_sensitive_test",
    "val": "_sensitive_val",
    "all": "_sensitive_all",
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


def resolve_runtime_files(
    kwargs: dict[str, Any],
    files: Mapping[str, Any] | None = None,
    *,
    legacy_keys: tuple[str, ...] = ("data_file", "score_file", "metadata_file"),
) -> DataFiles:
    """Resolve canonical runtime files payload from explicit and legacy kwargs.

    This helper pops legacy flat file kwargs from ``kwargs`` and merges them with
    an optional ``files`` mapping into canonical ``DataFiles``.
    """
    files_payload = files if isinstance(files, Mapping) else None
    legacy_files = {key: kwargs.pop(key) for key in legacy_keys if key in kwargs}
    return merge_data_files(
        files_payload,
        legacy_files if len(legacy_files) > 0 else None,
    )


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


def normalize_runtime_split_mode(
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    default: str = "test",
) -> str:
    """Normalize stage/split aliases to canonical runtime split tokens."""
    token = str(mode or default).strip().lower()
    alias_map = dict(_RUNTIME_SPLIT_ALIASES)
    if aliases:
        alias_map.update(
            {
                str(key).strip().lower(): str(value).strip().lower()
                for key, value in aliases.items()
            },
        )
    resolved = alias_map.get(token, token)
    if resolved not in _SPLIT_DATA_ATTRS:
        raise ValueError(
            f"Unknown runtime split mode '{mode}'. Expected one of {sorted(_SPLIT_DATA_ATTRS)}",
        )
    return resolved


def resolve_data_split_payload(
    data: Any,
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    fallback_to_all: bool = False,
) -> tuple[Any, Any]:
    """Resolve ``(y, X)`` payload for a runtime split mode from a data object."""
    if data is None:
        return None, None
    resolved_mode = normalize_runtime_split_mode(mode, aliases=aliases)
    y_attr, x_attr = _SPLIT_DATA_ATTRS[resolved_mode]
    y = getattr(data, y_attr, None)
    X = getattr(data, x_attr, None)
    if fallback_to_all and resolved_mode != "all":
        if y is None:
            y = getattr(data, "_y", None)
        if X is None:
            X = getattr(data, "_X", None)
    return y, X


def resolve_sensitive_split_payload(
    data: Any,
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    fallback_to_all: bool = False,
) -> Any:
    """Resolve sensitive-feature payload for a runtime split mode."""
    if data is None:
        return None
    resolved_mode = normalize_runtime_split_mode(mode, aliases=aliases)
    sensitive_attr = _SPLIT_SENSITIVE_ATTRS[resolved_mode]
    sensitive = getattr(data, sensitive_attr, None)
    if sensitive is None:
        legacy_attr = sensitive_attr.removeprefix("_")
        sensitive = getattr(data, legacy_attr, None)
    if sensitive is None and fallback_to_all and resolved_mode != "all":
        all_attr = _SPLIT_SENSITIVE_ATTRS["all"]
        sensitive = getattr(data, all_attr, None)
        if sensitive is None:
            sensitive = getattr(data, all_attr.removeprefix("_"), None)
    return sensitive


@dataclass(eq=False, kw_only=True)
class DataPluginRuntimeMixin(RuntimeBase):
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

@dataclass(eq=False, kw_only=True)
class ScoringOrchestratorMixin(OrchestratorBase, DataPluginRuntimeMixin):
    """Backward-compatible alias wrapper for centralized plugin orchestration."""

    default_stage: Final[str] = DEFAULT_DATA_SCORE_STAGE
    stage_aliases: ClassVar[dict[str, str]] = _STAGE_ALIASES
    mode_aliases: ClassVar[dict[str, str]] = _MODE_ALIASES
    score_stage_aliases: ClassVar[dict[str, str]] = _STAGE_ALIASES
    score_stage_order: ClassVar[tuple[str, ...]] = tuple(
        stage for stage in CANONICAL_DATA_STAGES if stage not in {"all", "auto"}
    )
    score_event_aliases: ClassVar[dict[str, str]] = _EVENT_ALIASES
    score_stage_to_hook: ClassVar[dict[str, str]] = {
        "pre-load": "before_load_data",
        "pre-sample": "before_sample",
        "post-sample": "after_sample",
        "post-pipeline": "after_pipeline",
    }
    _score_orchestration_active: bool = True

    def _normalize_score_mode(self, mode: str) -> str:
        return normalize_data_score_mode(mode)

    def _stage_hook_token(self, stage: str) -> str:
        return stage_hook_token(stage)


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
    "normalize_runtime_split_mode",
    "resolve_data_split_payload",
    "resolve_sensitive_split_payload",
    "merge_data_files",
    "merge_files",
    "resolve_runtime_files",
    "DataPluginRuntimeMixin",
    "ScoringOrchestratorMixin",
]
