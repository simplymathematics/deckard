"""Canonical model runtime contract helpers.

This module defines the implementation-level contract for model runtimes and
provides helpers to normalize timing/runtime state in a consistent shape.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, TypedDict

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
    aliases = {
        "train": "train",
        "training": "train",
        "test": "test",
        "eval": "test",
        "evaluation": "test",
        "val": "val",
        "valid": "val",
        "validation": "val",
    }
    if token in aliases:
        return aliases[token]
    raise ValueError(
        "Unknown model score mode "
        f"'{mode}'. Must be one of {list(CANONICAL_MODEL_SCORE_MODES)}",
    )


def ensure_model_runtime_contract(target: Any) -> Any:
    """Populate canonical runtime attributes on a ModelConfig-like object."""
    if not hasattr(target, "score_dict") or getattr(target, "score_dict") is None:
        target.score_dict = {}

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
