import deckard.score.base as score_base
from deckard.detector.canon import CANONICAL_DETECTOR_SCORE_STAGE_ALIASES
from deckard.score import canon as score_canon


def test_score_base_reexports_canon_stage_mode_contracts_by_identity() -> None:
    assert score_base.SUPPORTED_SCORING_STAGES is score_canon.SUPPORTED_SCORING_STAGES
    assert (
        score_base.SUPPORTED_DATA_SCORE_MODES is score_canon.SUPPORTED_DATA_SCORE_MODES
    )
    assert (
        score_base.SUPPORTED_MODEL_SCORE_MODES
        is score_canon.SUPPORTED_MODEL_SCORE_MODES
    )
    assert (
        score_base.SUPPORTED_EXPERIMENT_DEFENSE_SCORING_STAGES
        is score_canon.SUPPORTED_EXPERIMENT_DEFENSE_SCORING_STAGES
    )
    assert (
        score_base.SUPPORTED_ATTACK_SCORE_MODES
        is score_canon.SUPPORTED_ATTACK_SCORE_MODES
    )
    assert (
        score_base.SUPPORTED_EXPERIMENT_SCORE_MODES
        is score_canon.SUPPORTED_EXPERIMENT_SCORE_MODES
    )
    assert (
        score_base.SUPPORTED_DETECTOR_SCORE_MODES
        is score_canon.SUPPORTED_DETECTOR_SCORE_MODES
    )
    assert (
        score_base.SUPPORTED_PIPELINE_SCORE_MODES
        is score_canon.SUPPORTED_PIPELINE_SCORE_MODES
    )
    assert (
        score_base.DEFAULT_SCORING_MODE_BY_TYPE
        is score_canon.DEFAULT_SCORING_MODE_BY_TYPE
    )
    assert (
        score_base.DEFAULT_SCORING_STAGE_BY_TYPE
        is score_canon.DEFAULT_SCORING_STAGE_BY_TYPE
    )
    assert (
        score_base.SCORING_STAGE_TOKEN_ALIASES
        is score_canon.SCORING_STAGE_TOKEN_ALIASES
    )


def test_detector_and_score_detector_stage_vocabularies_are_consistent() -> None:
    detector_stage_family = {
        stage
        for stage in set(CANONICAL_DETECTOR_SCORE_STAGE_ALIASES.values())
        if stage in {"pre-filter", "post-filter", "val"}
    }
    score_stage_family = {
        score_canon.ScoringDetectorStage.PRE_FILTER.value,
        score_canon.ScoringDetectorStage.POST_FILTER.value,
        score_canon.ScoringDetectorStage.VAL_FILTER.value,
    }
    assert detector_stage_family == score_stage_family
