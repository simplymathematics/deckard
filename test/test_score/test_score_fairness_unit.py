from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deckard.score.base import ScorerConfig, ScorerDictConfig
from deckard.score.fairness import (
    FairlearnScoreDictConfig,
    _flatten_metric_frame_by_group,
    _group_metric_difference,
    _resolve_sensitive_features,
    _resolve_sensitive_from_kwargs_or_data,
    _series_like_to_float_dict,
    fairness_demographic_parity_difference,
    fairness_equalized_odds_difference,
    fairness_group_mean_prediction_difference,
)


def test_resolve_sensitive_features_modes_and_errors():
    data = SimpleNamespace(
        _sensitive_train=np.array([0, 1]),
        _sensitive_test=np.array([1, 1]),
        _sensitive_val=np.array([0, 0]),
        _sensitive_all=np.array([0, 1]),
    )

    assert _resolve_sensitive_features(None, [0, 1], mode="test") is None
    assert _resolve_sensitive_features(data, [0, 1], mode="train") is not None
    assert _resolve_sensitive_features(data, [0, 1], mode="val") is not None
    assert _resolve_sensitive_features(data, [0, 1], mode="all") is not None
    assert _resolve_sensitive_features(data, [0], mode="test") is None

    with pytest.raises(ValueError, match="Unsupported fairness scoring mode"):
        _resolve_sensitive_features(data, [0, 1], mode="bad")


def test_fairness_core_scorers_require_sensitive_features():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError, match="sensitive_features are required"):
        fairness_demographic_parity_difference(y_true=y_true, y_pred=y_pred, data=None)

    with pytest.raises(ValueError, match="sensitive_features are required"):
        fairness_equalized_odds_difference(y_true=y_true, y_pred=y_pred, data=None)

    with pytest.raises(ValueError, match="sensitive_features are required"):
        _resolve_sensitive_from_kwargs_or_data(y_true=y_true, data=None)


def test_flatten_helpers_and_scalar_series_like_branches():
    by_group = pd.DataFrame({"metric": [0.1, 0.2]}, index=["A", "B"])
    flat = _flatten_metric_frame_by_group(by_group)
    assert flat["A_metric"] == 0.1

    assert _series_like_to_float_dict(0.5) == {"value": 0.5}


def test_fairlearn_score_dict_post_init_branches_and_type_errors():
    nested = ScorerDictConfig(
        scorers={
            "acc": ScorerConfig(score_name="acc", score_function="sklearn.metrics.accuracy_score"),
        },
    )
    cfg = FairlearnScoreDictConfig(
        group_scorers={
            "nested": nested,
            "callable": lambda y_true, y_pred, **kwargs: 1.0,
            "string": "sklearn.metrics.accuracy_score",
        },
    )
    # After normalization, nested ScorerDictConfig keys are flattened
    assert "acc" in cfg.group_scorers
    assert "callable" in cfg.group_scorers
    assert "string" in cfg.group_scorers

    with pytest.raises(TypeError, match="must contain a dict"):
        FairlearnScoreDictConfig(group_scorers=[{"group_scorers": "bad"}])

    with pytest.raises(TypeError, match="must be ScorerConfig"):
        FairlearnScoreDictConfig(group_scorers={"bad": 7})

    # New: Expect ValueError if both group_scorers and scorers are empty
    with pytest.raises(ValueError, match="group_scorers must not be empty"):
        FairlearnScoreDictConfig()


def test_build_metric_frame_import_error_and_call_validation(monkeypatch):
    cfg = FairlearnScoreDictConfig(
        group_scorers={
            "acc": ScorerConfig(score_name="acc", score_function="sklearn.metrics.accuracy_score"),
        },
    )

    monkeypatch.setattr("deckard.score.fairness.MetricFrame", None)
    with pytest.raises(ImportError, match="optional dependency"):
        cfg._build_metric_frame(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            sensitive_features=np.array([0, 1]),
        )

    with pytest.raises(ValueError, match="sensitive_features are required"):
        cfg(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            mode="test",
            data=None,
        )


def test_call_overall_value_branch_and_invalid_group_reduction(monkeypatch):
    cfg = FairlearnScoreDictConfig(
        group_scorers={
            "m1": ScorerConfig(score_name="m1", score_function="sklearn.metrics.accuracy_score"),
            "m2": ScorerConfig(score_name="m2", score_function="sklearn.metrics.accuracy_score"),
        },
        include_group_overall=True,
        include_group_by_group=False,
        group_reduction="none",
    )

    class FakeMetricFrame:
        overall = 0.25
        by_group = {"A": {"m1": 0.5}}

        def difference(self, method=None):
            _ = method
            return pd.Series({"m1": 0.1})

        def ratio(self, method=None):
            _ = method
            return pd.Series({"m1": 0.9})

    monkeypatch.setattr(cfg, "_build_metric_frame", lambda **kwargs: FakeMetricFrame())

    out = cfg(
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        sensitive_features=np.array(["A", "B"]),
        mode=None,
    )
    assert out["m1_overall"] == 0.25
    assert out["m2_overall"] == 0.25

    cfg_bad = FairlearnScoreDictConfig(
        group_scorers={
            "m1": ScorerConfig(score_name="m1", score_function="sklearn.metrics.accuracy_score"),
        },
        group_reduction="invalid",  # type: ignore[arg-type]
    )
    monkeypatch.setattr(cfg_bad, "_build_metric_frame", lambda **kwargs: FakeMetricFrame())
    with pytest.raises(ValueError, match="group_reduction must be one of"):
        cfg_bad(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            sensitive_features=np.array(["A", "B"]),
            mode="test",
        )


def test_group_difference_and_group_mean_edge_branches_with_nan_groups():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])

    assert _group_metric_difference(y_true, y_pred, np.array([1, 1]), lambda a, b: float(np.mean(np.abs(a - b)))) == 0.0

    # np.nan yields an all-False equality mask branch for one group.
    mixed_groups = np.array([np.nan, 1.0], dtype=float)
    diff = _group_metric_difference(
        y_true,
        y_pred,
        mixed_groups,
        lambda a, b: float(np.mean(np.abs(a - b))),
    )
    assert diff == 0.0

    mean_diff_single = fairness_group_mean_prediction_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=np.array([1, 1]),
    )
    assert mean_diff_single == 0.0

    mean_diff_nan = fairness_group_mean_prediction_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=mixed_groups,
    )
    assert mean_diff_nan == 0.0
