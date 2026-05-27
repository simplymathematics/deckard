"""Canonical data runtime stage, mode, timing, and persistence helpers."""

from __future__ import annotations

from typing import Any, Final, Mapping, TypedDict

from ..artifacts import ScoreDict
from ..orchestration import (
    CANONICAL_RUNTIME_METHODS,
    DEFAULT_SCORE_MODE,
    DEFAULT_SCORE_STAGE,
    MODE_ALIASES as _MODE_ALIASES,
    STAGE_ALIASES as _STAGE_ALIASES,
    DataRuntimeStateMixin as DataPluginRuntimeMixin,
    ScoreOrchestratorMixin as ScoringOrchestratorMixin,
    normalize_runtime_split_mode as _normalize_runtime_split_mode,
    resolve_data_split_payload as _resolve_data_split_payload,
    resolve_sensitive_split_payload as _resolve_sensitive_split_payload,
    stage_hook_token as _stage_hook_token,
)

CANONICAL_DATA_METHODS: Final[tuple[str, ...]] = CANONICAL_RUNTIME_METHODS

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

DEFAULT_DATA_SCORE_STAGE: Final[str] = DEFAULT_SCORE_STAGE
DEFAULT_DATA_SCORE_MODE: Final[str] = DEFAULT_SCORE_MODE


class BaseFiles(TypedDict, total=False):
    """Typed mapping for shared data runtime file keys.

    Attributes:
        params_file: Optional parameter/config artifact path.
    """

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
        target.score_dict = ScoreDict()
    else:
        target.score_dict = ScoreDict.from_payload(getattr(target, "score_dict"))

    for field in CANONICAL_DATA_RUNTIME_FIELDS:
        if field in {"score_dict", "files", "times"}:
            continue
        if not hasattr(target, field):
            setattr(target, field, None)
    return target


def stage_hook_token(stage: str) -> str:
    """Convert canonical stage names into hook-safe tokens."""
    return _stage_hook_token(stage)


def normalize_data_score_mode(mode: str) -> str:
    """Normalize score split mode names to canonical tokens."""
    key = str(mode).strip().lower().replace(" ", "-")
    if key in _MODE_ALIASES:
        return _MODE_ALIASES[key]
    raise ValueError(
        f"Unknown data score mode '{mode}'. Must be one of {list(CANONICAL_DATA_SCORE_MODES)}",
    )


def normalize_data_score_stage(value: str) -> str:
    """Normalize score stage aliases to canonical stage tokens."""
    key = str(value).strip().lower().replace(" ", "-")
    if key in {"all", "auto"}:
        return key
    if key in _STAGE_ALIASES:
        return _STAGE_ALIASES[key]
    raise ValueError(
        f"Unknown data score stage '{value}'. Must be one of {list(CANONICAL_DATA_STAGES)}",
    )


def normalize_runtime_split_mode(
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    default: str = "test",
) -> str:
    """Normalize stage/split aliases to canonical runtime split tokens."""
    return _normalize_runtime_split_mode(mode, aliases=aliases, default=default)


def resolve_data_split_payload(
    data: Any,
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    fallback_to_all: bool = False,
) -> tuple[Any, Any]:
    """Resolve ``(y, X)`` payload for a runtime split mode from a data object."""
    return _resolve_data_split_payload(
        data,
        mode,
        aliases=aliases,
        fallback_to_all=fallback_to_all,
    )


def resolve_sensitive_split_payload(
    data: Any,
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    fallback_to_all: bool = False,
) -> Any:
    """Resolve sensitive-feature payload for a runtime split mode."""
    return _resolve_sensitive_split_payload(
        data,
        mode,
        aliases=aliases,
        fallback_to_all=fallback_to_all,
    )


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
