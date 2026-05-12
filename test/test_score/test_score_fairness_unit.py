import pytest
import numpy as np
import pandas as pd
from deckard.score import fairness
from deckard.score.fairness import (
    as_group_scorer,
    _resolve_sensitive_features,
    fairness_demographic_parity_difference,
    fairness_equalized_odds_difference,
    fairness_group_mean_prediction_difference,
    fairness_group_mae_difference,
    fairness_group_mse_difference,
    _flatten_metric_frame_by_group,
    _series_like_to_float_dict,
    FairlearnScoreDictConfig,
    DefaultFairlearnClassificationConfig,
    DefaultFairlearnRegressionConfig,
)
from deckard.score.base import ScorerConfig, ScorerDictConfig


class DummyData:
    def __init__(
        self,
        sensitive_train=None,
        sensitive_test=None,
        sensitive_val=None,
        sensitive_all=None,
    ):
        self._sensitive_train = sensitive_train
        self._sensitive_test = sensitive_test
        self._sensitive_val = sensitive_val
        self._sensitive_all = sensitive_all


# --- as_group_scorer ---
def test_as_group_scorer_with_dict():
    scorer_dict = {"accuracy": ScorerConfig("accuracy", lambda y_true, y_pred: 1.0)}
    group = as_group_scorer(scorer_dict)
    assert isinstance(group, FairlearnScoreDictConfig)
    assert group.scorers["accuracy"].score_name == "accuracy"


def test_as_group_scorer_with_scorerdictconfig():
    scorer_dict = ScorerDictConfig(
        scorers={"accuracy": ScorerConfig("accuracy", lambda y_true, y_pred: 1.0)},
    )
    group = as_group_scorer(scorer_dict)
    assert isinstance(group, FairlearnScoreDictConfig)
    assert group.scorers["accuracy"].score_name == "accuracy"


@pytest.mark.parametrize("bad_input", [None, 123, "foo"])
def test_as_group_scorer_typeerror(bad_input):
    with pytest.raises(TypeError):
        as_group_scorer(bad_input)


# --- _resolve_sensitive_features ---
def test_resolve_sensitive_features_modes():
    arr = np.array([0, 1, 0, 1])
    data = DummyData(
        sensitive_train=arr,
        sensitive_test=arr,
        sensitive_val=arr,
        sensitive_all=arr,
    )
    y = np.ones(4)
    assert np.all(_resolve_sensitive_features(data, y, mode="train") == arr)
    assert np.all(_resolve_sensitive_features(data, y, mode="test") == arr)
    assert np.all(_resolve_sensitive_features(data, y, mode="attack") == arr)
    assert np.all(_resolve_sensitive_features(data, y, mode="val") == arr)
    assert np.all(_resolve_sensitive_features(data, y, mode="attack-val") == arr)
    assert np.all(_resolve_sensitive_features(data, y, mode="all") == arr)
    with pytest.raises(ValueError):
        _resolve_sensitive_features(data, y, mode="badmode")


def test_resolve_sensitive_features_shape_mismatch():
    arr = np.array([0, 1, 0])
    data = DummyData(sensitive_train=arr)
    y = np.ones(4)
    assert _resolve_sensitive_features(data, y, mode="train") is None


def test_resolve_sensitive_features_none():
    data = DummyData()
    y = np.ones(4)
    assert _resolve_sensitive_features(data, y, mode="train") is None


# --- fairness_demographic_parity_difference & fairness_equalized_odds_difference ---
def test_fairness_demographic_parity_difference_and_equalized_odds(monkeypatch):
    # Patch fairlearn.metrics functions
    monkeypatch.setattr(
        fairness,
        "demographic_parity_difference",
        lambda **kwargs: 0.5,
    )
    monkeypatch.setattr(fairness, "equalized_odds_difference", lambda **kwargs: 0.2)
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    sensitive = [0, 1, 0, 1]
    # Should work with direct sensitive_features
    assert (
        fairness_demographic_parity_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive,
        )
        == 0.5
    )
    assert (
        fairness_equalized_odds_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive,
        )
        == 0.2
    )
    # Should work with data object
    data = DummyData(sensitive_test=sensitive)
    assert fairness_demographic_parity_difference(y_true, y_pred, data=data) == 0.5
    assert fairness_equalized_odds_difference(y_true, y_pred, data=data) == 0.2
    # Should raise if sensitive_features missing
    monkeypatch.setattr(
        fairness,
        "demographic_parity_difference",
        lambda **kwargs: 0.5,
    )
    with pytest.raises(ValueError):
        fairness_demographic_parity_difference(y_true, y_pred)


# --- _flatten_metric_frame_by_group & _series_like_to_float_dict ---
def test_flatten_metric_frame_by_group():
    df = pd.DataFrame({"accuracy": [0.8, 0.9], "f1": [0.7, 0.8]}, index=["A", "B"])
    flat = _flatten_metric_frame_by_group(df)
    assert flat == {"A_accuracy": 0.8, "A_f1": 0.7, "B_accuracy": 0.9, "B_f1": 0.8}


def test_series_like_to_float_dict():
    s = pd.Series([1.0, 2.0], index=["a", "b"])
    assert _series_like_to_float_dict(s) == {"a": 1.0, "b": 2.0}
    df = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
    assert _series_like_to_float_dict(df) == {"a_x": 1.0, "b_x": 2.0}
    assert _series_like_to_float_dict(3.5) == {"value": 3.5}


# --- group mean/mae/mse difference ---
def test_fairness_group_mean_prediction_difference():
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]
    sensitive = [0, 1, 0, 1]
    result = fairness_group_mean_prediction_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive,
    )
    assert abs(result - 0.7) < 1e-6


def test_fairness_group_mae_difference():
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]
    sensitive = [0, 1, 0, 1]
    result = fairness_group_mae_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive,
    )
    assert abs(result) < 1e-6


def test_fairness_group_mse_difference():
    y_true = [0, 1, 0, 1]
    y_pred = [0.1, 0.9, 0.2, 0.8]
    sensitive = [0, 1, 0, 1]
    result = fairness_group_mse_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive,
    )
    assert abs(result) < 1e-6


# --- DefaultFairlearnClassificationConfig & DefaultFairlearnRegressionConfig ---
def test_default_fairlearn_classification_config():
    cfg = DefaultFairlearnClassificationConfig()
    assert cfg.classifier is True
    assert "accuracy" in cfg.scorers
    assert isinstance(cfg, FairlearnScoreDictConfig)


def test_default_fairlearn_regression_config():
    cfg = DefaultFairlearnRegressionConfig()
    assert cfg.classifier is False
    assert "mse" in cfg.scorers
    assert isinstance(cfg, FairlearnScoreDictConfig)
