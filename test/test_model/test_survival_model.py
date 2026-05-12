from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from lifelines.exceptions import ConvergenceError

from deckard.model.survival import SurvivalModelConfig


def test_survival_model_post_init_normalizes_classifier_and_target():
    cfg = SurvivalModelConfig(classifier=True)

    assert cfg.classifier is False
    assert cfg._target_ == "deckard.model.SurvivalModelConfig"
    assert isinstance(cfg.score_dict, dict)


def test_initialize_aft_fitter_valid_and_invalid_model_type():
    fitter = SurvivalModelConfig._initialize_aft_fitter("weibull", {})
    assert fitter.__class__.__name__ == "WeibullAFTFitter"

    with pytest.raises(ValueError, match="Model type unknown not recognized"):
        SurvivalModelConfig._initialize_aft_fitter("unknown", {})


def test_clean_data_for_aft_raises_when_target_missing():
    df = pd.DataFrame({"x": [1, 2], "group": ["a", "b"]})

    with pytest.raises(ValueError, match="Target adv_failure_rate not in dataframe"):
        SurvivalModelConfig.clean_data_for_aft(
            data=df,
            covariate_list=["x", "group"],
            target="adv_failure_rate",
        )


def test_clean_data_for_aft_filters_sentinels_and_applies_dummy_dict():
    df = pd.DataFrame(
        {
            "x": [1.0, -1e10, 3.0, 4.0],
            "group": ["a", "b", "a", "b"],
            "adv_failure_rate": [0.1, 0.2, 0.3, 1e10],
        },
    )

    cleaned = SurvivalModelConfig.clean_data_for_aft(
        data=df,
        covariate_list=["x", "group"],
        target="adv_failure_rate",
        dummy_dict={"group": "Group"},
    )

    assert "adv_failure_rate" in cleaned.columns
    assert "Group a" in cleaned.columns or "Group b" in cleaned.columns
    assert np.all(cleaned["x"].to_numpy() != -1e10)
    assert np.all(cleaned["adv_failure_rate"].to_numpy() != 1e10)


def test_clean_data_for_aft_auto_encodes_object_columns_without_dummy_dict():
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
            "cat": ["x", "y", "x"],
            "adv_failure_rate": [0.1, 0.2, 0.3],
        },
    )

    cleaned = SurvivalModelConfig.clean_data_for_aft(
        data=df,
        covariate_list=["feature", "cat"],
        target="adv_failure_rate",
    )

    assert "cat" not in cleaned.columns
    assert "x" in cleaned.columns or "y" in cleaned.columns
    assert all(dtype.kind in {"f", "i", "u"} for dtype in cleaned.dtypes)


def test_make_survival_model_table_writes_csv_and_restores_t0(monkeypatch, tmp_path):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E", t0=0.5)
    original_t0 = cfg.t0
    x_test = pd.DataFrame({"T": [1.0, 2.0, 3.0], "E": [1, 0, 1]})

    def _calibration(**kwargs):
        _ = kwargs
        return None, 0.11, 0.22

    monkeypatch.setattr(cfg, "survival_probability_calibration", _calibration)

    fitter = SimpleNamespace(AIC_=10.0, BIC_=20.0, concordance_index_=0.75)
    table = cfg.make_survival_model_table(
        models={"weibull": fitter, "skip": None},
        dataset="toy",
        X_train=x_test,
        X_test=x_test,
        folder=str(tmp_path),
        t0s={"weibull": 0.8},
    )

    assert not table.empty
    assert "AIC" in table.columns
    assert "BIC" in table.columns
    assert "concordance" in table.columns
    assert "ICI" in table.columns
    assert "E50" in table.columns
    assert (tmp_path / "aft_comparison.csv").exists()
    assert cfg.t0 == original_t0


def test_score_uses_custom_scorer_directly():
    cfg = SurvivalModelConfig()
    cfg.scorer = lambda **kwargs: {"custom": 1.0}

    out = cfg._score(y_true=pd.DataFrame(), y_pred=object())

    assert out == {"custom": 1.0}


def test_score_collects_concordance_and_handles_calibration_failure(monkeypatch):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E")
    cfg.scorer = None
    y_true = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0]})
    model = SimpleNamespace(concordance_index_=0.9)

    def _raise_calibration(**kwargs):
        _ = kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(cfg, "survival_probability_calibration", _raise_calibration)

    out = cfg._score(y_true=y_true, y_pred=model)

    assert out["concordance"] == 0.9
    assert "ici" not in out
    assert "e50" not in out


def test_initialize_aft_fitter_sets_aalen_default_alpha():
    fitter = SurvivalModelConfig._initialize_aft_fitter("aalen", {})
    assert fitter.__class__.__name__ == "AalenAdditiveFitter"


def test_fit_aft_validates_required_columns():
    cfg = SurvivalModelConfig(
        duration_col="T", event_col="E", survival_model="weibull"
    )

    with pytest.raises(ValueError, match="Column T not found in data"):
        cfg.fit_aft(pd.DataFrame({"E": [1, 0]}))

    with pytest.raises(ValueError, match="Column E not found in data"):
        cfg.fit_aft(pd.DataFrame({"T": [1.0, 2.0]}))


def test_fit_aft_retries_on_convergence_nan_delta(monkeypatch):
    cfg = SurvivalModelConfig(
        duration_col="T", event_col="E", survival_model="weibull"
    )
    calls = {"n": 0}

    class FakeFitter:
        summary = pd.DataFrame({"x": [1]})

        def fit(self, df, **kwargs):
            _ = df
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConvergenceError("delta contains nan value(s)")
            assert "fit_options" in kwargs
            return self

    monkeypatch.setattr(cfg, "_initialize_aft_fitter", lambda **_: FakeFitter())

    df = pd.DataFrame({"T": [1.0, 2.0, 3.0], "E": [1, 0, 1]})
    model = cfg.fit_aft(df)

    assert isinstance(model, FakeFitter)
    assert calls["n"] == 2


def test_fit_aft_retries_with_slsqp_for_other_errors(monkeypatch):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E", survival_model="aalen")
    calls = {"n": 0}

    class FakeFitter:
        summary = pd.DataFrame({"x": [1]})

        def fit(self, df, **kwargs):
            _ = (df, kwargs)
            calls["n"] += 1
            if calls["n"] == 1:
                raise AttributeError("different failure")
            return self

    fitter = FakeFitter()
    monkeypatch.setattr(cfg, "_initialize_aft_fitter", lambda **_: fitter)

    df = pd.DataFrame({"T": [1.0, 2.0, 3.0], "E": [1, 0, 1]})
    cfg.fit_aft(df)

    assert calls["n"] == 2
    assert getattr(fitter, "_scipy_fit_method") == "SLSQP"


def test_score_handles_concordance_property_errors_and_nan_calibration(monkeypatch):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E")
    cfg.scorer = None
    y_true = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0]})

    class BrokenConcordance:
        @property
        def concordance_index_(self):
            raise RuntimeError("broken concordance")

    monkeypatch.setattr(
        cfg,
        "survival_probability_calibration",
        lambda **_: (None, np.nan, np.nan),
    )

    out = cfg._score(y_true=y_true, y_pred=BrokenConcordance())

    assert out == {}


def test_clean_data_for_aft_raises_when_target_removed_by_dummy_encoding():
    df = pd.DataFrame(
        {
            "feat": [1.0, 2.0],
            "adv_failure_rate": ["low", "high"],
        },
    )

    with pytest.raises(
        ValueError, match="Target adv_failure_rate not in cleaned dataframe"
    ):
        SurvivalModelConfig.clean_data_for_aft(
            data=df,
            covariate_list=["feat"],
            target="adv_failure_rate",
            dummy_dict={"adv_failure_rate": "AFR"},
        )


def test_survival_probability_calibration_crc_fit_failure_returns_nan_curve(
    monkeypatch,
):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E", t0=0.5)

    class FakeCensoringType:
        @staticmethod
        def is_right_censoring(model):
            _ = model
            return True

        @staticmethod
        def is_left_censoring(model):
            return False

        @staticmethod
        def is_interval_censoring(model):
            return False

    class FailingCRC:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def fit_right_censoring(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("fail fit")

    class FakeModel:
        duration_col = "T"
        event_col = "E"

        def predict_survival_function(self, df, times):
            _ = times
            return pd.DataFrame([[0.9] for _ in range(len(df))], index=df.index)

    monkeypatch.setattr("deckard.model.survival.CensoringType", FakeCensoringType)
    monkeypatch.setattr("deckard.model.survival.CRCSplineFitter", FailingCRC)

    df = pd.DataFrame({"T": [1.0, 2.0, 3.0], "E": [1, 0, 1]})
    _, ici, e50, curve = cfg.survival_probability_calibration(
        model=FakeModel(),
        df=df,
        return_curve=True,
        plot=False,
    )

    assert np.isnan(ici)
    assert np.isnan(e50)
    assert curve["observed"].isna().all()


def test_survival_probability_calibration_left_censor_and_delta_fallback(monkeypatch):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E", t0=0.5)

    class FakeCensoringType:
        @staticmethod
        def is_right_censoring(model):
            return False

        @staticmethod
        def is_left_censoring(model):
            return True

        @staticmethod
        def is_interval_censoring(model):
            return False

    class CRCForLeft:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            self.calls = 0

        def fit_left_censoring(self, *args, **kwargs):
            _ = (args, kwargs)

        def predict_survival_function(self, df, times):
            _ = times
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("delta failure")
            return pd.DataFrame([[0.85] for _ in range(len(df))], index=df.index)

    class FakeModel:
        duration_col = "T"
        event_col = "E"

        def predict_survival_function(self, df, times):
            _ = times
            return pd.DataFrame([[0.9] for _ in range(len(df))], index=df.index)

    monkeypatch.setattr("deckard.model.survival.CensoringType", FakeCensoringType)
    monkeypatch.setattr("deckard.model.survival.CRCSplineFitter", CRCForLeft)

    df = pd.DataFrame({"T": [1.0, 2.0, 3.0], "E": [1, 0, 1]})
    _, ici, e50 = cfg.survival_probability_calibration(
        model=FakeModel(),
        df=df,
        return_curve=False,
        plot=True,
    )

    assert np.isnan(ici)
    assert np.isnan(e50)


def test_survival_probability_calibration_interval_and_default_fit_paths(monkeypatch):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E", t0=0.5)

    class FakeModel:
        duration_col = "T"
        event_col = "E"

        def predict_survival_function(self, df, times):
            _ = times
            return pd.DataFrame([[0.9] for _ in range(len(df))], index=df.index)

    df = pd.DataFrame({"T": [1.0, 2.0, 3.0], "E": [1, 0, 1]})

    class IntervalCensoring:
        @staticmethod
        def is_right_censoring(model):
            return False

        @staticmethod
        def is_left_censoring(model):
            return False

        @staticmethod
        def is_interval_censoring(model):
            return True

    class DefaultCensoring:
        @staticmethod
        def is_right_censoring(model):
            return False

        @staticmethod
        def is_left_censoring(model):
            return False

        @staticmethod
        def is_interval_censoring(model):
            return False

    class CRCWithBranchMarkers:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)
            self.mode = None

        def fit_interval_censoring(self, *args, **kwargs):
            _ = (args, kwargs)
            self.mode = "interval"

        def fit(self, *args, **kwargs):
            _ = (args, kwargs)
            self.mode = "default"

        def predict_survival_function(self, df, times):
            _ = times
            return pd.DataFrame([[0.8] for _ in range(len(df))], index=df.index)

    monkeypatch.setattr("deckard.model.survival.CensoringType", IntervalCensoring)
    monkeypatch.setattr("deckard.model.survival.CRCSplineFitter", CRCWithBranchMarkers)
    _, ici_interval, e50_interval = cfg.survival_probability_calibration(
        model=FakeModel(),
        df=df,
        plot=False,
    )

    monkeypatch.setattr("deckard.model.survival.CensoringType", DefaultCensoring)
    monkeypatch.setattr("deckard.model.survival.CRCSplineFitter", CRCWithBranchMarkers)
    _, ici_default, e50_default = cfg.survival_probability_calibration(
        model=FakeModel(),
        df=df,
        plot=False,
    )

    assert np.isfinite(ici_interval)
    assert np.isfinite(e50_interval)
    assert np.isfinite(ici_default)
    assert np.isfinite(e50_default)


def test_make_survival_model_table_handles_metric_attribute_and_calibration_failures(
    monkeypatch, tmp_path
):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E", t0=0.5)
    x_test = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0]})

    class NoisyFitter:
        def __getattribute__(self, name):
            if name in {"AIC_", "AIC_partial_", "BIC_", "concordance_index_"}:
                raise RuntimeError("metric access failed")
            return super().__getattribute__(name)

    monkeypatch.setattr(
        cfg,
        "survival_probability_calibration",
        lambda **_: (_ for _ in ()).throw(RuntimeError("calibration failed")),
    )

    table = cfg.make_survival_model_table(
        models={"bad": NoisyFitter()},
        dataset="toy",
        X_train=x_test,
        X_test=x_test,
        folder=str(tmp_path),
    )

    assert not table.empty
    assert "model" in table.columns
    assert "t0" in table.columns


def test_make_survival_model_table_returns_empty_when_all_models_none(tmp_path):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E", t0=0.5)
    x_test = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0]})

    table = cfg.make_survival_model_table(
        models={"skip": None},
        dataset="toy",
        X_train=x_test,
        X_test=x_test,
        folder=str(tmp_path),
    )

    assert table.empty
    assert not (tmp_path / "aft_comparison.csv").exists()
