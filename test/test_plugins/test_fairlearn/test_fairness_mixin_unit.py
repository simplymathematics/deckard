from types import SimpleNamespace
import logging

import numpy as np
import pandas as pd
import pytest


from deckard.plugins.fairlearn.model import (
    _SensitiveBehaviorMixin,
)

# Use the logger from conftest_logging.py
logger = logging.getLogger(__name__)

pytest.importorskip("fairlearn")


class _DummyMixin(_SensitiveBehaviorMixin):
    pass


def _dummy_data():
    return SimpleNamespace(
        X_train=pd.DataFrame({"x": [1, 2]}, index=[0, 1]),
        X_test=pd.DataFrame({"x": [3, 4]}, index=[10, 11]),
        _X=pd.DataFrame({"x": [5, 6]}, index=[20, 21]),
        y_train=pd.Series([0, 1]),
        _sensitive_train=pd.Series(["a", "b"]),
        _sensitive_test=pd.Series(["a", "b"]),
        _sensitive_all=pd.Series(["a", "b"]),
    )


def test_runtime_sensitive_source_and_split_resolution_errors():
    d = _DummyMixin()
    d.data = _dummy_data()

    assert list(d._resolve_runtime_sensitive_source("train")) == ["a", "b"]
    assert list(d._resolve_runtime_sensitive_source("test")) == ["a", "b"]
    assert list(d._resolve_runtime_sensitive_source("all")) == ["a", "b"]

    with pytest.raises(NotImplementedError):
        d._resolve_runtime_sensitive_source("val")
    with pytest.raises(ValueError):
        d._resolve_runtime_sensitive_source("bad")

    assert d._resolve_scoring_split("train") == "train"
    assert d._resolve_scoring_split("test") == "test"
    assert d._resolve_scoring_split("attack") == "test"
    assert d._resolve_scoring_split("all") == "all"
    with pytest.raises(NotImplementedError):
        d._resolve_scoring_split("val")
    with pytest.raises(ValueError):
        d._resolve_scoring_split("unknown")


def test_validate_sensitive_series_checks_empty_null_blank():
    d = _DummyMixin()

    assert d._validate_sensitive_series(None, "ctx") is None

    with pytest.raises(ValueError, match="empty"):
        d._validate_sensitive_series([], "ctx")
    with pytest.raises(ValueError, match="all null"):
        d._validate_sensitive_series([None, np.nan], "ctx")
    with pytest.raises(ValueError, match="blank"):
        d._validate_sensitive_series([" ", ""], "ctx")


def test_infer_and_resolve_sensitive_features_for_batch_paths(monkeypatch):
    d = _DummyMixin()
    d.data = _dummy_data()

    # Test all valid scoring modes
    assert d._infer_split_from_batch(d.data.X_train, scoring_mode="train") == "train"
    assert (
        d._infer_split_from_batch(d.data.X_test.copy(), scoring_mode="test") == "test"
    )
    # For 'val', just call and check result (should not raise)
    result_val = d._infer_split_from_batch(d.data.X_train, scoring_mode="val")
    assert result_val == "val"
    # For 'all', should work if implemented, else raise if not supported
    try:
        result = d._infer_split_from_batch(d.data.X_train, scoring_mode="all")
        assert result == "all"
    except NotImplementedError:
        pass

    batch = pd.DataFrame({"x": [1, 2]})
    assert d._resolve_sensitive_features_for_batch(batch, split="train") is not None

    d.data._sensitive_train = pd.Series(["a"])  # length mismatch
    assert d._resolve_sensitive_features_for_batch(batch, split="train") is None

    d.data._sensitive_train = pd.Series(["a", "b"])
    monkeypatch.setattr(
        pd.Series,
        "reindex",
        lambda self, idx: (_ for _ in ()).throw(RuntimeError("reindex fail")),
    )
    assert d._resolve_sensitive_features_for_batch(batch, split="train") is None


def test_method_signature_detection_and_optional_sensitive_calling():
    d = _DummyMixin()

    def with_sensitive(x, sensitive_features=None):
        return (x, sensitive_features)

    def with_kwargs(x, **kwargs):
        return (x, kwargs.get("sensitive_features"))

    def plain(x):
        return x

    assert d._method_accepts_sensitive_features(with_sensitive)
    assert d._method_accepts_sensitive_features(with_kwargs)
    assert not d._method_accepts_sensitive_features(plain)

    assert d._call_with_optional_sensitive(with_sensitive, 1, "s") == (1, "s")
    assert d._call_with_optional_sensitive(plain, 1, "s") == 1


def test_fit_defended_estimator_paths():
    d = _DummyMixin()
    data = _dummy_data()

    class FitWithSensitive:
        def __init__(self):
            self.calls = []

        def fit(self, x, y, sensitive_features=None):
            self.calls.append((x, y, sensitive_features))
            return self

    class FitPlain:
        def __init__(self):
            self.calls = []

        def fit(self, x, y):
            self.calls.append((x, y))
            return self

    f1 = FitWithSensitive()
    out1 = d._fit_defended_estimator(f1, data)
    assert out1 is f1
    assert len(f1.calls) == 1
    assert f1.calls[0][2] is not None

    f2 = FitPlain()
    out2 = d._fit_defended_estimator(f2, data)
    assert out2 is f2
    assert len(f2.calls) == 1

    sentinel = object()
    assert d._fit_defended_estimator(sentinel, None) is sentinel
