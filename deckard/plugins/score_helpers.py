"""Shared helpers for plugin score-tail merge behavior."""

from __future__ import annotations

from typing import Any

from ..artifacts import ScoreDict


def merge_prefixed_tail_scores(
    tail_scores: dict[str, Any],
    *,
    existing_scores: dict[str, Any] | None,
    prefix: str,
) -> ScoreDict:
    """Merge tail scores and prefix only keys that collide with existing scores."""
    existing = dict(existing_scores or {})
    if len(existing) == 0:
        return ScoreDict.from_payload(tail_scores)

    merged_tail: dict[str, Any] = {}
    for key, value in tail_scores.items():
        if key in existing:
            merged_tail[f"{prefix}_{key}"] = value
        else:
            merged_tail[key] = value
    return ScoreDict.from_payload(merged_tail)
