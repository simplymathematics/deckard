"""Canonical model runtime contract helpers.

This module defines the implementation-level contract for model runtimes and
provides helpers to normalize timing/runtime state in a consistent shape.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, TypedDict

from ..artifacts import ScoreDict
from ..orchestration import (
    MODE_ALIASES as _MODE_ALIASES,
    STAGE_ALIASES as _STAGE_ALIASES,
    normalize_runtime_split_mode as _normalize_runtime_split_mode,
)

CANONICAL_MODEL_METHODS: Final[tuple[str, ...]] = (
    "initialize_model",
    "train",
    "score",
    "load_score_times",
    "load_cached_predictions",
    "train_or_load_model",
    "evaluate_model",
    "persist_outputs",
    "__call__",
)

CANONICAL_MODEL_SCORE_MODES: Final[tuple[str, ...]] = (
    "train",
    "test",
    "val",
)

CANONICAL_MODEL_SCORE_STAGES: Final[tuple[str, ...]] = (
    "pre-load",
    "pre-sample",
    "post-sample",
    "post-pipeline",
    "pre-defense",
    "post-defense",
    "all",
    "auto",
)

CANONICAL_MODEL_SCORE_STAGE_ALIASES: Final[dict[str, str]] = {
    **dict(_STAGE_ALIASES),
    "pre-defense": "pre-defense",
    "pre_defense": "pre-defense",
    "predefense": "pre-defense",
    "before-defense": "pre-defense",
    "before_defense": "pre-defense",
    "post-defense": "post-defense",
    "post_defense": "post-defense",
    "postdefense": "post-defense",
    "after-defense": "post-defense",
    "after_defense": "post-defense",
}

CANONICAL_MODEL_SCORE_MODE_ALIASES: Final[dict[str, str]] = dict(_MODE_ALIASES)

CANONICAL_MODEL_RUNTIME_SPLIT_ALIASES: Final[dict[str, str]] = {
    **CANONICAL_MODEL_SCORE_MODE_ALIASES,
    "auto": "test",
}

CANONICAL_MODEL_TIMES: Final[tuple[str, ...]] = (
    "training_time",
    "prediction_time",
    "training_prediction_time",
    "prediction_score_time",
    "training_score_time",
)

CANONICAL_MODEL_RUNTIME_FIELDS: Final[tuple[str, ...]] = (
    "_model",
    "score_dict",
    "training_predictions",
    "predictions",
    "val_predictions",
    "training_probabilities",
    "probabilities",
    "val_probabilities",
    "training_time",
    "prediction_time",
    "val_prediction_time",
    "training_prediction_time",
    "training_score_time",
    "prediction_score_time",
    "val_score_time",
    "defense_application_time",
    "training_n",
    "prediction_n",
    "val_n",
)

DEFAULT_MODEL_SCORE_MODE: Final[str] = "test"

CANONICAL_MODEL_TRAINER_ALIASES: Final[tuple[str, ...]] = (
    "sklearn",
    "pretrained",
    "partial_fit",
    "partial_fit_pruning",
    "pruning",
    "pytorch",
)

CANONICAL_MODEL_DEFENSE_STAGES: Final[tuple[str, ...]] = (
    "pre_art_defense",
    "pre_fit",
    "post_fit_pre_predict",
)

CANONICAL_MODEL_DEFAULT_STAGE: Final[str] = "post-predict"


class ModelFiles(TypedDict, total=False):
    """Canonical model persistence aliases used by ModelConfig.__call__."""

    model_file: str | None
    test_predictions_file: str | None
    training_predictions_file: str | None
    train_predictions_file: str | None
    training_probabilities_file: str | None
    test_probabilities_file: str | None
    score_file: str | None


class DefenseFiles(TypedDict, total=False):
    """Canonical defense persistence aliases used by defense runtimes.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    defended_model_file: str | None
    defended_predictions_file: str | None
    defended_probabilities_file: str | None
    score_file: str | None


class ModelTimes(TypedDict, total=False):
    """Canonical model timing keys (plus optional extensions)."""

    training_time: float | None
    prediction_time: float | None
    training_prediction_time: float | None
    prediction_score_time: float | None
    training_score_time: float | None


def ensure_canonical_model_times(
    times: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an extensible timing dict with canonical model keys present."""
    merged = {key: None for key in CANONICAL_MODEL_TIMES}
    if times:
        merged.update(dict(times))
    return merged


def normalize_model_score_mode(mode: str | None) -> str:
    """Normalize model score mode aliases to canonical split-scoped names."""
    token = str(mode or DEFAULT_MODEL_SCORE_MODE).strip().lower()
    aliases = CANONICAL_MODEL_SCORE_MODE_ALIASES
    if token in aliases:
        return aliases[token]
    raise ValueError(
        "Unknown model score mode "
        f"'{mode}'. Must be one of {list(CANONICAL_MODEL_SCORE_MODES)}",
    )


def normalize_model_score_stage(stage: str | None) -> str:
    """Normalize model score stage aliases to canonical orchestration stages."""
    token = str(stage or "post-pipeline").strip().lower().replace(" ", "-")
    if token in {"all", "auto"}:
        return token
    resolved = CANONICAL_MODEL_SCORE_STAGE_ALIASES.get(token)
    if resolved is None:
        raise ValueError(
            "Unknown model score stage "
            f"'{stage}'. Must be one of {list(CANONICAL_MODEL_SCORE_STAGES)}",
        )
    return resolved


def normalize_model_runtime_split_mode(
    mode: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
    default: str = "test",
) -> str:
    """Normalize model split aliases to canonical runtime split tokens."""
    merged_aliases = dict(CANONICAL_MODEL_RUNTIME_SPLIT_ALIASES)
    if aliases:
        merged_aliases.update(
            {
                str(key).strip().lower(): str(value).strip().lower()
                for key, value in aliases.items()
            },
        )
    return _normalize_runtime_split_mode(mode, aliases=merged_aliases, default=default)


def ensure_model_runtime_contract(target: Any) -> Any:
    """Populate canonical runtime attributes on a ModelConfig-like object."""
    target.score_dict = ScoreDict.from_payload(
        getattr(target, "score_dict", None) or {},
    )

    for field in CANONICAL_MODEL_RUNTIME_FIELDS:
        if field == "score_dict":
            continue
        if not hasattr(target, field):
            setattr(target, field, None)

    return target


def normalize_model_trainer_alias(alias: str | None) -> str:
    """Normalize trainer alias tokens for model runtime trainer composition."""
    token = str(alias or "sklearn").strip().lower().replace("-", "_")
    aliases = {
        "sklearn": "sklearn",
        "base": "sklearn",
        "default": "sklearn",
        "pretrained": "pretrained",
        "pre_trained": "pretrained",
        "cache": "pretrained",
        "partial_fit": "partial_fit",
        "partialfit": "partial_fit",
        "partial_fit_pruning": "partial_fit_pruning",
        "partialfit_pruning": "partial_fit_pruning",
        "partial_prune": "partial_fit_pruning",
        "pruning": "pruning",
        "prune": "pruning",
        "pytorch": "pytorch",
        "torch": "pytorch",
    }
    if token in aliases:
        return aliases[token]
    raise ValueError(
        "Unknown model trainer alias "
        f"'{alias}'. Must be one of {list(CANONICAL_MODEL_TRAINER_ALIASES)}",
    )


def resolve_model_defense_stage(
    defense_name: str | None,
    *,
    default_stage: str = "post_fit_pre_predict",
) -> str:
    """Resolve canonical model defense stage from a defense class path token."""
    token = str(defense_name or "").strip().lower()
    if token == "":
        return default_stage

    if token.startswith("anjana.") or ".anjana." in token:
        return "pre_art_defense"

    if token.startswith("fairlearn.reductions"):
        return "pre_fit"

    if token.startswith("fairlearn.adversarial"):
        return "post_fit_pre_predict"

    if token.startswith("fairlearn.postprocessing"):
        return "post_fit_pre_predict"

    return default_stage


def defense_stage_priority(stage: str | None) -> int:
    """Return ordering priority for model defense application stages."""
    token = str(stage or "post_fit_pre_predict").strip().lower()
    order = {
        "pre_art_defense": 0,
        "pre_fit": 1,
        "post_fit_pre_predict": 2,
    }
    return order.get(token, 99)
