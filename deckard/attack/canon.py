"""Canonical attack runtime contract helpers."""

from __future__ import annotations

from typing import Any, Final, TypedDict

from ..orchestration import MODE_ALIASES as _MODE_ALIASES

from ..artifacts import ScoreDict


class AttackFiles(TypedDict, total=False):
    """Canonical attack persistence aliases used by attack runtimes."""

    attack_file: str | None
    attack_predictions_file: str | None
    score_file: str | None


ATTACK_RUNTIME_STAGE_ALIASES: dict[str, str] = {
    "before_attack": "pre-attack",
    "before-attack": "pre-attack",
    "pre_attack": "pre-attack",
    "pre-attack": "pre-attack",
    "after_attack": "post-attack",
    "after-attack": "post-attack",
    "post_attack": "post-attack",
    "post-attack": "post-attack",
}

ATTACK_RUNTIME_VALID_MODES = frozenset({"auto", "train", "test", "val"})
ATTACK_RUNTIME_TIME_KEYS = (
    "attack_generation_time",
    "attack_prediction_time",
    "attack_score_time",
)

CANONICAL_ATTACK_SCORE_STAGES: Final[tuple[str, ...]] = (
    "pre-attack",
    "post-attack",
    "all",
    "auto",
)

CANONICAL_ATTACK_SCORE_STAGE_ALIASES: Final[dict[str, str]] = {
    "pre-attack": "pre-attack",
    "pre_attack": "pre-attack",
    "before-attack": "pre-attack",
    "before_attack": "pre-attack",
    "post-attack": "post-attack",
    "post_attack": "post-attack",
    "after-attack": "post-attack",
    "after_attack": "post-attack",
}

CANONICAL_ATTACK_SCORE_MODES: Final[tuple[str, ...]] = (
    "train",
    "test",
    "val",
    "all",
    "auto",
)

CANONICAL_ATTACK_SCORE_MODE_ALIASES: Final[dict[str, str]] = dict(_MODE_ALIASES)


def normalize_attack_stage(stage: str | None) -> str | None:
    """Normalize attack stage tokens into canonical hook stage names."""
    token = str(stage or "post-attack").strip().lower().replace("_", "-")
    if token == "auto":
        return None
    return ATTACK_RUNTIME_STAGE_ALIASES.get(token, token)


def normalize_attack_mode(mode: Any) -> str:
    """Normalize attack split mode and validate canonical values."""
    token = str(mode or "auto").strip().lower()
    if token not in ATTACK_RUNTIME_VALID_MODES:
        expected = ", ".join(sorted(ATTACK_RUNTIME_VALID_MODES))
        raise ValueError(
            f"Unsupported attack mode '{mode}'. Expected one of: {expected}.",
        )
    return token


def normalize_attack_score_stage(stage: str | None) -> str:
    """Normalize attack score stage aliases to canonical attack score stages."""
    token = str(stage or "post-attack").strip().lower().replace("_", "-")
    if token in {"all", "auto"}:
        return token
    resolved = CANONICAL_ATTACK_SCORE_STAGE_ALIASES.get(token)
    if resolved is None:
        raise ValueError(
            "Unknown attack score stage "
            f"'{stage}'. Must be one of {list(CANONICAL_ATTACK_SCORE_STAGES)}",
        )
    return resolved


def normalize_attack_score_mode(mode: str | None) -> str:
    """Normalize attack score mode aliases to canonical split tokens."""
    token = str(mode or "test").strip().lower().replace(" ", "-")
    resolved = CANONICAL_ATTACK_SCORE_MODE_ALIASES.get(token)
    if resolved is None:
        raise ValueError(
            "Unknown attack score mode "
            f"'{mode}'. Must be one of {list(CANONICAL_ATTACK_SCORE_MODES)}",
        )
    return resolved


def ensure_attack_runtime_contract(runtime: Any) -> None:
    """Ensure core attack runtime fields exist and are initialized."""
    if not hasattr(runtime, "score_dict") or runtime.score_dict is None:
        runtime.score_dict = ScoreDict()

    for attr in (
        "attack_time",
        "attack_prediction_time",
        "attack_score_time",
        "attack_predictions",
        "attacked_labels",
        "score_y_pred",
        "score_y_proba",
        "attack",
    ):
        if not hasattr(runtime, attr):
            setattr(runtime, attr, None)
