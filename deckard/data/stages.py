"""Canonical data scoring stage helpers.

This module defines the stage vocabulary used by data-family runtimes and
provides normalization utilities so scoring and hook dispatch stay consistent.
"""

from __future__ import annotations

from typing import Final

CANONICAL_DATA_SCORE_STAGES: Final[tuple[str, ...]] = (
    "pre-sample",
    "train",
    "test",
    "val",
    "post-sample",
    "post-pipeline",
    "all",
)

# Non-canonical names map to canonical stage routing.
DATA_SCORE_STAGE_ALIASES: Final[dict[str, str]] = {
    "pre-defense": "test",
    "post-defense": "test",
    "attack": "test",
    "attack-val": "val",
    "adversarial": "test",
    "benign": "test",
}


def normalize_data_score_stage(
    mode: str | None,
    *,
    default: str = "test",
) -> str:
    """Normalize runtime score mode to a canonical data score stage."""
    token = str(mode or default).strip().lower()
    token = DATA_SCORE_STAGE_ALIASES.get(token, token)
    if token not in CANONICAL_DATA_SCORE_STAGES:
        raise ValueError(
            f"DataConfig score_mode '{token}' not in {set(CANONICAL_DATA_SCORE_STAGES)}",
        )
    return token


def stage_hook_token(stage: str) -> str:
    """Return a hook-safe token for a canonical stage name."""
    return str(stage).strip().lower().replace("-", "_")
