import numpy as np
import pandas as pd
import pytest

import deckard.score as score_module
from deckard.score._runtime import resolve_yt_yp, series_like_to_float_dict
from deckard.score.canon import normalize_scorer_mode, normalize_stage_tokens


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


def test_series_like_to_float_dict_supports_tuple_index_dataframe_and_callable_values():
    frame = pd.DataFrame(
        {"score": [0.1, 0.2]},
        index=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")]),
    )
    assert series_like_to_float_dict(frame) == {
        "a_x_score": 0.1,
        "b_y_score": 0.2,
    }

    def callback():
        return 1.0

    out = series_like_to_float_dict({"hook": callback})
    assert out["hook"] is callback


def test_series_like_to_float_dict_rejects_non_scalar_arrays():
    with pytest.raises(TypeError, match="must be scalar"):
        series_like_to_float_dict(np.array([1.0, 2.0]))


def test_normalize_scorer_mode_uses_canonical_modes():
    assert normalize_scorer_mode(None) == "test"
    assert normalize_scorer_mode(" Train ") == "train"
    with pytest.raises(KeyError, match="Unsupported scoring mode"):
        normalize_scorer_mode("post-defense")


def test_normalize_stage_tokens_supports_strings_sequences_and_scalars():
    assert normalize_stage_tokens("train, Test ,") == {"train", "test"}
    assert normalize_stage_tokens(["attack", ["val", "attack-val"]]) == {
        "attack",
        "val",
        "attack-val",
    }
    assert normalize_stage_tokens(None) == set()
    assert normalize_stage_tokens(3) == {"3"}


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


def test_resolve_yt_yp_covers_train_val_attack_val_and_existing_predictions():
    data = _DummyData()
    model = _DummyModel()
    attack = _DummyAttack()

    y_true, y_pred = resolve_yt_yp("train", data, model, None, None, None)
    assert np.array_equal(y_true, data.y_train)
    assert np.array_equal(y_pred, model.training_predictions)

    y_true, y_pred = resolve_yt_yp("val", data, model, None, None, None)
    assert np.array_equal(y_true, data.y_val)
    assert np.array_equal(y_pred, model.val_predictions)

    y_true, y_pred = resolve_yt_yp("attack-val", data, None, attack, None, None)
    assert np.array_equal(y_true, data.y_val)
    assert np.array_equal(y_pred, attack.attack_predictions)

    y_true, y_pred = resolve_yt_yp(
        "test",
        data,
        model,
        None,
        np.array([9]),
        np.array([8]),
    )
    assert np.array_equal(y_true, np.array([8]))
    assert np.array_equal(y_pred, np.array([9]))


@pytest.mark.parametrize(
    ("wrapper_name", "loader_name", "symbol_name"),
    [
        (
            "fairness_demographic_parity_difference",
            "_load_fairlearn_score_symbol",
            "fairness_demographic_parity_difference",
        ),
        (
            "anjana_k_anonymity_score",
            "_load_anjana_score_symbol",
            "anjana_k_anonymity_score",
        ),
        (
            "survival_concordance_score",
            "_load_lifelines_score_symbol",
            "survival_concordance_score",
        ),
    ],
)
def test_score_wrappers_delegate_to_lazy_symbol_loaders(
    monkeypatch,
    wrapper_name,
    loader_name,
    symbol_name,
):
    def _loader(requested_name):
        assert requested_name == symbol_name

        def _impl(*args, **kwargs):
            return requested_name, args, kwargs

        return _impl

    monkeypatch.setattr(score_module, loader_name, _loader)
    wrapper = getattr(score_module, wrapper_name)
    result = wrapper(1, key=2)
    assert result == (symbol_name, (1,), {"key": 2})


def test_score_getattr_lazy_loads_optional_symbols(monkeypatch):
    sentinel = object()

    def _load_fairlearn():
        setattr(score_module, "DefaultFairlearnScorerDictConfig", sentinel)
        return True

    monkeypatch.setattr(score_module, "_load_fairlearn_score_symbols", _load_fairlearn)
    assert score_module.__getattr__("DefaultFairlearnScorerDictConfig") is sentinel

    with pytest.raises(AttributeError):
        score_module.__getattr__("definitely_missing_symbol")
