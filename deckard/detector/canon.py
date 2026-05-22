"""Canonical detector runtime contract helpers."""

from __future__ import annotations

from typing import Any

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


def normalize_detector_stage(stage: str | None) -> str:
    """Normalize detector stage tokens into canonical hook stage names."""
    token = str(stage or "post-detect").strip().lower().replace("_", "-")
    return DETECTOR_RUNTIME_STAGE_ALIASES.get(token, token)


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
