import numpy as np
import pandas as pd
import pytest

from deckard.score._runtime import resolve_yt_yp, series_like_to_float_dict
from deckard.score.canon import normalize_scorer_mode


class _DummyData:
    def __init__(self):
        self.y_test = np.array([1, 0])
        self.y_train = np.array([0, 1])
        self.y_val = np.array([1, 1])
        self._X = np.array([[1.0], [2.0]])
        self._y = np.array([1, 0])


class _DummyModel:
    def __init__(self):
        self.test_predictions = np.array([1, 1])
        self.training_predictions = np.array([0, 0])
        self.val_predictions = np.array([1, 0])


class _DummyAttack:
    def __init__(self):
        self.attack_predictions = np.array([0])
        self.attack_size = 1


def test_series_like_to_float_dict_dataframe_series_and_scalar():
    frame = pd.DataFrame({"score": [0.1, 0.2]}, index=["a", "b"])
    assert series_like_to_float_dict(frame) == {"a_score": 0.1, "b_score": 0.2}

    series = pd.Series([1.0, 2.0], index=["x", "y"])
    assert series_like_to_float_dict(series) == {"x": 1.0, "y": 2.0}

    assert series_like_to_float_dict(3.5) == {"value": 3.5}


def test_series_like_to_float_dict_supports_nested_dict_payloads():
    payload = {
        "summary": {"mean": np.float32(0.2), "max": np.float64(1.0)},
        "count": 3,
    }
    out = series_like_to_float_dict(payload)
    assert out["summary_mean"] == pytest.approx(0.2)
    assert out["summary_max"] == pytest.approx(1.0)
    assert out["count"] == pytest.approx(3.0)


def test_series_like_to_float_dict_rejects_non_scalar_arrays():
    with pytest.raises(TypeError, match="must be scalar"):
        series_like_to_float_dict(np.array([1.0, 2.0]))


def test_normalize_scorer_mode_uses_canonical_modes():
    assert normalize_scorer_mode(None) == "test"
    assert normalize_scorer_mode(" Train ") == "train"
    with pytest.raises(KeyError, match="Unsupported scoring mode"):
        normalize_scorer_mode("post-defense")


def test_resolve_yt_yp_uses_runtime_context_by_mode():
    data = _DummyData()
    model = _DummyModel()
    attack = _DummyAttack()

    y_true, y_pred = resolve_yt_yp("test", data, model, None, None, None)
    assert np.array_equal(y_true, data.y_test)
    assert np.array_equal(y_pred, model.test_predictions)

    y_true, y_pred = resolve_yt_yp("attack", data, None, attack, None, None)
    assert np.array_equal(y_true, np.array([1]))
    assert np.array_equal(y_pred, attack.attack_predictions)

    y_true, y_pred = resolve_yt_yp("pre-sample", data, None, None, None, None)
    assert np.array_equal(y_true, data._y)
    assert np.array_equal(y_pred, data._X)
