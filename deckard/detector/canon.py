"""Canonical detector runtime contract helpers."""

from __future__ import annotations

from typing import Any, Final, Mapping, TypedDict

from ..orchestration import (
    MODE_ALIASES as _MODE_ALIASES,
    STAGE_ALIASES as _STAGE_ALIASES,
    normalize_runtime_split_mode as _normalize_runtime_split_mode,
)


class DetectorFiles(TypedDict, total=False):
    """Canonical detector persistence aliases used by detector runtimes."""

    detector_model_file: str | None
    detected_predictions_file: str | None
    detected_probabilities_file: str | None
    score_file: str | None


DETECTOR_RUNTIME_STAGE_ALIASES: dict[str, str] = {
    "before_fit": "pre-fit",
    "before-fit": "pre-fit",
    "pre_fit": "pre-fit",
    "pre-fit": "pre-fit",
    "after_fit": "post-fit",
    "after-fit": "post-fit",
    "post_fit": "post-fit",
    "post-fit": "post-fit",
    "before_detect": "pre-detect",
    "before-detect": "pre-detect",
    "pre_detect": "pre-detect",
    "pre-detect": "pre-detect",
    "after_detect": "post-detect",
    "after-detect": "post-detect",
    "post_detect": "post-detect",
    "post-detect": "post-detect",
}

DETECTOR_RUNTIME_TIME_KEYS = (
    "detector_training_time",
    "detector_detection_time",
)

CANONICAL_DETECTOR_SCORE_STAGES: Final[tuple[str, ...]] = (
    "pre-load",
    "pre-sample",
    "post-sample",
    "post-pipeline",
    "all",
    "auto",
)

CANONICAL_DETECTOR_SCORE_STAGE_ALIASES: Final[dict[str, str]] = dict(_STAGE_ALIASES)

CANONICAL_DETECTOR_SCORE_MODES: Final[tuple[str, ...]] = (
    "train",
    "test",
    "val",
    "all",
)

CANONICAL_DETECTOR_SCORE_MODE_ALIASES: Final[dict[str, str]] = dict(_MODE_ALIASES)

CANONICAL_DETECTOR_RUNTIME_SPLIT_ALIASES: Final[dict[str, str]] = {
    **CANONICAL_DETECTOR_SCORE_MODE_ALIASES,
    "auto": "test",
    "detect": "test",
    "filter": "test",
}


def normalize_detector_stage(stage: str | None) -> str:
    """Normalize detector stage tokens into canonical hook stage names."""
    token = str(stage or "post-detect").strip().lower().replace("_", "-")
    return DETECTOR_RUNTIME_STAGE_ALIASES.get(token, token)


def normalize_detector_score_stage(stage: str | None) -> str:
    """Normalize detector score stage aliases to canonical orchestration stages."""
    token = str(stage or "post-pipeline").strip().lower().replace(" ", "-")
    if token in {"all", "auto"}:
        return token
    resolved = CANONICAL_DETECTOR_SCORE_STAGE_ALIASES.get(token)
    if resolved is None:
        raise ValueError(
            "Unknown detector score stage "
            f"'{stage}'. Must be one of {list(CANONICAL_DETECTOR_SCORE_STAGES)}",
        )
    return resolved


def normalize_detector_score_mode(mode: str | None) -> str:
    """Normalize detector score mode aliases to canonical split tokens."""
    token = str(mode or "test").strip().lower().replace(" ", "-")
    resolved = CANONICAL_DETECTOR_SCORE_MODE_ALIASES.get(token)
    if resolved is None:
        raise ValueError(
            "Unknown detector score mode "
            f"'{mode}'. Must be one of {list(CANONICAL_DETECTOR_SCORE_MODES)}",
        )
    return resolved


def normalize_detector_runtime_split_mode(
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    default: str = "test",
) -> str:
    """Normalize detector split aliases to canonical runtime split tokens."""
    merged_aliases = dict(CANONICAL_DETECTOR_RUNTIME_SPLIT_ALIASES)
    if aliases:
        merged_aliases.update(
            {
                str(key).strip().lower(): str(value).strip().lower()
                for key, value in aliases.items()
            },
        )
    return _normalize_runtime_split_mode(mode, aliases=merged_aliases, default=default)


def ensure_detector_runtime_contract(runtime: Any) -> None:
    """Ensure core detector runtime fields exist and are initialized."""
    if not hasattr(runtime, "score_dict") or runtime.score_dict is None:
        runtime.score_dict = {}

    for attr in (
        "detector",
        "detector_training_time",
        "detector_detection_time",
        "detector_predictions",
    ):
        if not hasattr(runtime, attr):
            setattr(runtime, attr, None)
