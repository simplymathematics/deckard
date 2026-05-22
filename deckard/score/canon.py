"""Canonical scorer runtime helpers and contract definitions."""

from __future__ import annotations

from typing import Any, TypedDict


CANON_SCORER_MODES: frozenset[str] = frozenset(
    {
        "train",
        "test",
        "val",
        "all",
        "attack",
        "attack-val",
        "pre-sample",
    },
)


class ScorerRuntimeContract(TypedDict, total=False):
    """Canonical scorer runtime payload contract.

    Keys are intentionally generic so model/data/attack/detector runtimes can
    compose into one scoring API without changing scorer call signatures.
    """

    score_mode: str
    stage: list[str]
    scores: dict[str, float | int]
    score_file: str


def normalize_scorer_mode(mode: str | None) -> str:
    """Normalize runtime score mode to canonical scorer scope tokens."""
    if mode is None:
        return "test"
    mode_token = str(mode).strip().lower()
    if mode_token in CANON_SCORER_MODES:
        return mode_token
    raise KeyError(f"Unsupported scoring mode '{mode}'.")


def normalize_stage_tokens(stage: Any) -> set[str]:
    """Normalize stage fields into lowercase token sets."""
    if stage is None:
        return set()
    if isinstance(stage, str):
        tokens = [token.strip().lower() for token in stage.split(",")]
        return {token for token in tokens if token != ""}
    if isinstance(stage, (list, tuple, set)):
        merged: set[str] = set()
        for item in stage:
            merged.update(normalize_stage_tokens(item))
        return merged
    return {str(stage).strip().lower()}
