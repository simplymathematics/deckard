from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deckard.data import DataConfig
from deckard.experiment.base import ExperimentConfig
from deckard.experiment.survival import SurvivalExperimentConfig
from deckard.model import ModelConfig


def _bare_instance():
    cfg = SurvivalExperimentConfig.__new__(SurvivalExperimentConfig)
    cfg.attack = None
    cfg.model = None
    cfg.duration_col = "T"
    cfg.data = DataConfig.__new__(DataConfig)
    return cfg


def test_post_init_validates_data_model_and_duration(monkeypatch):
    monkeypatch.setattr(ExperimentConfig, "__post_init__", lambda self: None)

    cfg = _bare_instance()
    cfg.data = None
    with pytest.raises(ValueError, match="requires a data config"):
        cfg.__post_init__()

    cfg = _bare_instance()
    cfg.data = "not-data-config"
    with pytest.raises(TypeError, match="Expected data to resolve to DataConfig"):
        cfg.__post_init__()

    cfg = _bare_instance()
    cfg.model = "bad-model"
    with pytest.raises(TypeError, match="Expected model to resolve to ModelConfig"):
        cfg.__post_init__()

    cfg = _bare_instance()
    cfg.duration_col = ""
    with pytest.raises(ValueError, match="duration_col must be provided"):
        cfg.__post_init__()


def test_attack_kind_and_candidate_metrics_misc_paths():
    assert SurvivalExperimentConfig._infer_attack_kind_from_label(np.nan) is None
    assert SurvivalExperimentConfig._infer_attack_kind_from_label("   ") is None

    metrics = SurvivalExperimentConfig._candidate_attack_metrics_for_kind(None)
    assert "evasion_success" in metrics
    assert "membership_inference_accuracy" in metrics
    assert "attribute_inference_accuracy" in metrics


def test_resolve_attack_size_from_uniform_column_without_row_index():
    df = pd.DataFrame({"attack_size": [5, 5, 5]})
    size = SurvivalExperimentConfig._resolve_attack_size(df)
    assert size == 5.0


def test_calculate_failures_under_attack_fallback_uses_attack_size_column():
    cfg = _bare_instance()
    df = pd.DataFrame(
        {
            "accuracy": [0.5, 0.5],
            "evasion_accuracy": [0.2, 0.4],
            "attack_size": [10.0, np.nan],
        },
    )

    out = cfg.calculate_failures_under_attack(df, attack_config=SimpleNamespace(attack_size=8, attack_kind="evasion"))

    assert "ben_failures" in out.columns
    assert "adv_failures" in out.columns
    assert np.isfinite(out["adv_failures"]).all()


def test_make_survival_model_table_handles_none_models_and_metric_failures(monkeypatch, tmp_path):
    cfg = _bare_instance()
    cfg.event_col = "E"

    x_test = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0]})

    empty = cfg.make_survival_model_table(
        models={"skip": None},
        dataset="toy",
        X_train=x_test,
        X_test=x_test,
        folder=str(tmp_path),
    )
    assert empty.empty

    class NoisyFitter:
        def __getattribute__(self, name):
            if name in {"AIC_partial_", "BIC_", "concordance_index_"}:
                raise RuntimeError("metric unavailable")
            if name == "AIC_":
                return 10.0
            return super().__getattribute__(name)

    import deckard.layers.survival as layer_survival

    monkeypatch.setattr(
        layer_survival,
        "survival_probability_calibration",
        lambda **kwargs: {"ICI": 0.1, "E50": 0.2},
    )

    table = cfg.make_survival_model_table(
        models={"weibull": NoisyFitter()},
        dataset="toy",
        X_train=x_test,
        X_test=x_test,
        folder=str(tmp_path),
        t0s={"weibull": 0.7},
    )

    assert not table.empty
    assert "AIC" in table.columns
    assert "ICI" in table.columns
    assert "E50" in table.columns
    assert (tmp_path / "aft_comparison.csv").exists()
