import importlib

import numpy as np
import pandas as pd
import pytest

import deckard.data.declarations as data_declarations
import deckard.score.data as score_data
import deckard.score.declarations as score_declarations


def test_score_declarations_loader_returns_when_examples_missing(
    tmp_path,
    monkeypatch,
):
    fake_file = tmp_path / "pkg" / "deckard" / "score" / "declarations.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("x=1")

    monkeypatch.setattr(score_declarations, "__file__", fake_file.as_posix())

    # Missing examples directory should be a no-op.
    score_declarations._load_example_score_configs()


def test_score_declarations_loader_skips_bad_yaml(tmp_path, monkeypatch):
    fake_file = tmp_path / "pkg" / "deckard" / "score" / "declarations.py"
    score_dir = tmp_path / "pkg" / "examples" / "sklearn" / "config" / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    (score_dir / "bad.yaml").write_text("broken: [")

    monkeypatch.setattr(score_declarations, "__file__", fake_file.as_posix())

    def _boom(_yaml_file):
        raise RuntimeError("bad yaml")

    monkeypatch.setattr(score_declarations.OmegaConf, "load", _boom)

    # Loader should swallow config parse failures.
    score_declarations._load_example_score_configs()


def test_data_declarations_register_sampler_configs_exception_path(monkeypatch):
    import deckard.data.sample as sample_mod

    calls = []

    def _record_call():
        calls.append("register")

    monkeypatch.setattr(
        sample_mod,
        "register_sampler_configs",
        _record_call,
    )
    reloaded = importlib.reload(data_declarations)

    assert reloaded is not None
    assert calls == []


def test_score_data_coerce_features_dataframe_series_and_vector():
    s = pd.Series([1.0, 2.0], name="named")
    out_series = score_data._coerce_features_dataframe(s)
    assert list(out_series.columns) == ["named"]

    out_vector = score_data._coerce_features_dataframe(np.array([1.0, 2.0]))
    assert list(out_vector.columns) == ["feature_0"]


def test_score_data_reference_resolution_reference_and_missing_column_error():
    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    ref, out_X = score_data._resolve_reference_vector([0, 1], X, reference=[9, 8])
    assert np.array_equal(ref, np.array([9, 8]))
    assert list(out_X.columns) == ["a", "b"]

    with pytest.raises(ValueError, match="reference_column 'nope' not found"):
        score_data._resolve_reference_vector([0, 1], X, reference_column="nope")


def test_score_data_is_discrete_reference_empty_and_object():
    assert score_data._is_discrete_reference(np.array([])) is True
    assert (
        score_data._is_discrete_reference(np.array(["x", "y"], dtype=object)) is True
    )


def test_score_data_mutual_information_raises_when_no_features_left():
    y = np.array([0, 1, 0, 1])
    X = pd.DataFrame({"label": y})

    with pytest.raises(ValueError, match="No feature columns available"):
        score_data._feature_mutual_information_vector(
            y_true=y,
            X=X,
            reference_column="label",
        )


def test_score_data_class_imbalance_ratio_empty_and_zero_min_count(monkeypatch):
    assert score_data.data_class_imbalance_ratio_score([], None) == 0.0

    original = score_data.pd.Series.value_counts

    def _fake_value_counts(self, dropna=False):
        _ = dropna
        return pd.Series([3.0, 0.0])

    monkeypatch.setattr(score_data.pd.Series, "value_counts", _fake_value_counts)
    try:
        assert score_data.data_class_imbalance_ratio_score([0, 1], None) == float(
            "inf",
        )
    finally:
        monkeypatch.setattr(score_data.pd.Series, "value_counts", original)


def test_score_data_empirical_cdf_empty_reference_raises():
    y = pd.Series([np.nan, np.nan])
    X = pd.DataFrame({"x": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Reference vector is empty"):
        score_data.data_empirical_cdf_function_score(
            y_true=y,
            X=X,
            reference=[np.nan, np.nan],
        )
