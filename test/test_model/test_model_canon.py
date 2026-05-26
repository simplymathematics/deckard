from types import SimpleNamespace

import pytest

from deckard.model.canon import (
    CANONICAL_MODEL_RUNTIME_FIELDS,
    CANONICAL_MODEL_SCORE_MODES,
    CANONICAL_MODEL_TIMES,
    defense_stage_priority,
    ensure_canonical_model_times,
    ensure_model_runtime_contract,
    normalize_model_score_mode,
    normalize_model_trainer_alias,
    resolve_model_defense_stage,
)


def test_model_canon_times_contains_required_keys():
    times = ensure_canonical_model_times()
    for key in CANONICAL_MODEL_TIMES:
        assert key in times


def test_model_canon_times_preserves_extensions():
    times = ensure_canonical_model_times(
        {
            "training_time": 1.2,
            "custom_plugin_time": 0.4,
        },
    )
    assert times["training_time"] == 1.2
    assert times["custom_plugin_time"] == 0.4


@pytest.mark.parametrize(
    "value,expected",
    [
        ("train", "train"),
        ("training", "train"),
        ("test", "test"),
        ("eval", "test"),
        ("val", "val"),
        ("validation", "val"),
        (None, "test"),
    ],
)
def test_model_canon_score_mode_normalization(value, expected):
    assert normalize_model_score_mode(value) == expected


def test_model_canon_score_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        normalize_model_score_mode("post-pipeline")


def test_model_canon_runtime_contract_populates_missing_fields():
    runtime = ensure_model_runtime_contract(SimpleNamespace(score_dict=None))
    assert isinstance(runtime.score_dict, dict)
    for field in CANONICAL_MODEL_RUNTIME_FIELDS:
        assert hasattr(runtime, field)


def test_model_canon_declares_split_scoped_score_modes_only():
    assert set(CANONICAL_MODEL_SCORE_MODES) == {"train", "test", "val"}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("default", "sklearn"),
        ("pre_trained", "pretrained"),
        ("partialfit", "partial_fit"),
        ("partial_prune", "partial_fit_pruning"),
        ("torch", "pytorch"),
    ],
)
def test_model_canon_trainer_alias_normalization(value, expected):
    assert normalize_model_trainer_alias(value) == expected


@pytest.mark.parametrize(
    "defense_name,expected",
    [
        ("anjana.defense.SomeDefense", "pre_art_defense"),
        ("fairlearn.reductions.ExponentiatedGradient", "pre_fit"),
        (
            "fairlearn.adversarial.AdversarialFairnessClassifier",
            "post_fit_pre_predict",
        ),
        ("fairlearn.postprocessing.ThresholdOptimizer", "post_fit_pre_predict"),
        ("art.defences.preprocessor.FeatureSqueezing", "post_fit_pre_predict"),
    ],
)
def test_model_canon_defense_stage_resolution(defense_name, expected):
    assert resolve_model_defense_stage(defense_name) == expected


def test_model_canon_defense_stage_priority_order():
    assert defense_stage_priority("pre_art_defense") < defense_stage_priority(
        "pre_fit"
    )
    assert defense_stage_priority("pre_fit") < defense_stage_priority(
        "post_fit_pre_predict"
    )
