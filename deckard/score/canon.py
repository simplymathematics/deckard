"""Canonical scorer runtime helpers and contract definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Final, TypedDict

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


class ScoringDefenseStage(str, Enum):
    """Enum representing the defense stage for scoring context."""

    PRE_DEFENSE = "pre-defense"
    POST_DEFENSE = "post-defense"
    VAL_DEFENSE = "val"


class ScoringPipelineStage(str, Enum):
    """Enum representing the pipeline stage for scoring context."""

    POST_PIPELINE = "post-pipeline"
    VAL_PIPELINE = "val"


class ScoringAttackStage(str, Enum):
    """Enum representing the attack stage for scoring context."""

    PRE_ATTACK = "benign"
    POST_ATTACK = "adversarial"
    VAL_ATTACK = "val"


class ScoringDataStage(str, Enum):
    """Enum representing the data transformation stage for scoring context."""

    PRE_SAMPLE = "pre-sample"
    POST_SAMPLE = "post-sample"
    VAL_ATTACK = "val"


class ScoringModelStage(str, Enum):
    """Enum representing the model stage for scoring context."""

    MODEL_TRAIN = "train"
    MODEL_TEST = "test"
    MODEL_VAL = "val"


class ScoringDetectorStage(str, Enum):
    """Enum representing the detector stage for scoring context."""

    PRE_FILTER = "pre-filter"
    POST_FILTER = "post-filter"
    VAL_FILTER = "val"


class ScoringDVCStage(str, Enum):
    """Enum representing DVC hook score stages for monitoring scorers."""

    DATA_SCORE = "data-score"
    MODEL_SCORE = "model-score"
    ATTACK_SCORE = "attack-score"
    DETECTOR_SCORE = "detector-score"


CANON_SCORING_STAGE_ENUMS: Final[tuple[type[Enum], ...]] = (
    ScoringDefenseStage,
    ScoringPipelineStage,
    ScoringAttackStage,
    ScoringDataStage,
    ScoringModelStage,
    ScoringDetectorStage,
    ScoringDVCStage,
)

SUPPORTED_SCORING_STAGES: Final[frozenset[str]] = frozenset(
    str(member.value).strip().lower()
    for enum_cls in CANON_SCORING_STAGE_ENUMS
    for member in enum_cls
)

SUPPORTED_DATA_SCORE_MODES: Final[frozenset[str]] = frozenset(
    {
        ScoringDataStage.PRE_SAMPLE.value,
        ScoringModelStage.MODEL_TRAIN.value,
        ScoringModelStage.MODEL_TEST.value,
        ScoringModelStage.MODEL_VAL.value,
    },
)

SUPPORTED_MODEL_SCORE_MODES: Final[frozenset[str]] = frozenset(
    {
        ScoringModelStage.MODEL_TRAIN.value,
        ScoringModelStage.MODEL_TEST.value,
        ScoringModelStage.MODEL_VAL.value,
    },
)

SUPPORTED_EXPERIMENT_DEFENSE_SCORING_STAGES: Final[frozenset[str]] = frozenset(
    {
        ScoringDefenseStage.PRE_DEFENSE.value,
        ScoringDefenseStage.POST_DEFENSE.value,
        ScoringDataStage.POST_SAMPLE.value,
    },
)

SUPPORTED_ATTACK_SCORE_MODES: Final[frozenset[str]] = frozenset(
    {
        ScoringAttackStage.PRE_ATTACK.value,
        ScoringAttackStage.POST_ATTACK.value,
        ScoringAttackStage.VAL_ATTACK.value,
    },
)

SUPPORTED_EXPERIMENT_SCORE_MODES: Final[frozenset[str]] = frozenset(
    {
        ScoringDefenseStage.PRE_DEFENSE.value,
        ScoringDefenseStage.POST_DEFENSE.value,
        ScoringDefenseStage.VAL_DEFENSE.value,
        ScoringPipelineStage.POST_PIPELINE.value,
        ScoringPipelineStage.VAL_PIPELINE.value,
        ScoringAttackStage.PRE_ATTACK.value,
        ScoringAttackStage.POST_ATTACK.value,
        ScoringAttackStage.VAL_ATTACK.value,
    },
)

SUPPORTED_DETECTOR_SCORE_MODES: Final[frozenset[str]] = frozenset(
    {
        ScoringDetectorStage.PRE_FILTER.value,
        ScoringDetectorStage.POST_FILTER.value,
        ScoringDetectorStage.VAL_FILTER.value,
    },
)

SUPPORTED_PIPELINE_SCORE_MODES: Final[frozenset[str]] = frozenset(
    {
        ScoringPipelineStage.POST_PIPELINE.value,
        ScoringPipelineStage.VAL_PIPELINE.value,
    },
)

DEFAULT_SCORING_MODE_BY_TYPE: Final[dict[str, str]] = {
    "data": "all",
    "model": "test",
    "attack": "test",
    "detector": "test",
}

DEFAULT_SCORING_STAGE_BY_TYPE: Final[dict[str, str]] = {
    "data": "post-pipeline",
    "model": "post-predict",
    "attack": "post-attack",
    "detector": "post-filter",
}

SCORING_STAGE_TOKEN_ALIASES: Final[dict[str, str]] = {
    "post-predict": ScoringDVCStage.MODEL_SCORE.value,
    "post_predict": ScoringDVCStage.MODEL_SCORE.value,
    "postpredict": ScoringDVCStage.MODEL_SCORE.value,
    "pre-attack": ScoringAttackStage.PRE_ATTACK.value,
    "pre_attack": ScoringAttackStage.PRE_ATTACK.value,
    "post-attack": ScoringAttackStage.POST_ATTACK.value,
    "post_attack": ScoringAttackStage.POST_ATTACK.value,
}


class ScorerRuntimeContract(TypedDict, total=False):
    """Canonical scorer runtime payload contract.

    Keys are intentionally generic so model/data/attack/detector runtimes can
    compose into one scoring API without changing scorer call signatures.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
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
