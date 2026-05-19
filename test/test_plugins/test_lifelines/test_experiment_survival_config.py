from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import pytest

from deckard.data import DataConfig
from deckard.experiment.base import ExperimentConfig
from deckard.plugins.lifelines.experiment import SurvivalExperimentConfig
from deckard.model import ModelConfig


class TestSurvivalExperimentConfig(unittest.TestCase):
    def test_allows_survival_only_config_without_attack(self):
        config = SurvivalExperimentConfig(
            data=DataConfig(dataset_name="make_regression", classifier=False),
            model="cox",
            target="E",
            classifier=False,
            duration_col="T",
            event_col="E",
        )
        self.assertIsInstance(config, SurvivalExperimentConfig)
        self.assertEqual(config.model, "cox")

    def test_requires_attack_when_aux_model_present(self):
        with self.assertRaises(ValueError):
            SurvivalExperimentConfig(
                data=DataConfig(
                    dataset_name="make_regression",
                    classifier=False,
                ),
                model="cox",
                target="E",
                classifier=False,
                aux_model=ModelConfig(
                    model_type="sklearn.linear_model.LogisticRegression",
                    classifier=True,
                    model_params={"max_iter": 10},
                ),
                duration_col="T",
                event_col="E",
            )

    def test_requires_data_config(self):
        with self.assertRaises(ValueError):
            SurvivalExperimentConfig(
                data=None,
                model="cox",
                target="E",
                duration_col="T",
                event_col="E",
                classifier=False,
            )

    def test_survival_config_initializes(self):
        config = SurvivalExperimentConfig(
            data=DataConfig(
                dataset_name="make_regression",
                classifier=False,
                target=None,
            ),
            model="cox",
            target="E",
            classifier=False,
            duration_col="T",
            event_col="E",
        )
        self.assertIsInstance(config, SurvivalExperimentConfig)
        self.assertEqual(config.model, "cox")


def _bare_instance():
    cfg = SurvivalExperimentConfig.__new__(SurvivalExperimentConfig)
    cfg.attack = None
    cfg.model = None
    cfg.target = "E"
    cfg.event_col = "E"
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
    cfg.model = ModelConfig(
        model_type="sklearn.tree.DecisionTreeClassifier",
        classifier=True,
        model_params={"max_depth": 1},
    )
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        cfg.__post_init__()

    cfg = _bare_instance()
    cfg.model = "weibull"
    cfg.duration_col = ""
    with pytest.raises(ValueError, match="duration_col must be a non-empty string"):
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

    out = cfg.calculate_failures_under_attack(
        df,
        attack_config=SimpleNamespace(attack_size=8, attack_kind="evasion"),
    )

    assert "ben_failures" in out.columns
    assert "adv_failures" in out.columns
    assert np.isfinite(out["adv_failures"]).all()


def test_make_survival_model_table_handles_none_models_and_metric_failures(
    monkeypatch,
    tmp_path,
):
    cfg = _bare_instance()
    cfg.event_col = "E"

    x_test = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0]})

    empty = cfg.make_survival_model_table(
        models={"skip": None},
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

    monkeypatch.setattr(
        SurvivalExperimentConfig,
        "survival_probability_calibration",
        staticmethod(lambda **kwargs: {"ICI": 0.1, "E50": 0.2}),
    )

    table = cfg.make_survival_model_table(
        models={"weibull": NoisyFitter()},
        X_test=x_test,
        folder=str(tmp_path),
        t0s={"weibull": 0.7},
    )

    assert not table.empty
    assert "AIC" in table.columns
    assert "ICI" in table.columns
    assert "E50" in table.columns
    assert (tmp_path / "aft_comparison.csv").exists()


def test_normalize_optional_dict_rejects_non_mapping():
    with pytest.raises(TypeError, match="must be a dict"):
        SurvivalExperimentConfig._normalize_optional_dict("plot", ["not", "a", "dict"])


def test_build_survival_plot_config_prefers_model_config():
    cfg = _bare_instance()
    cfg.model_config = {"cox": {"t0": 0.5}}
    out = cfg._build_survival_plot_config()
    assert out == {"cox": {"t0": 0.5}}


def test_prepare_loaded_data_raises_for_missing_fillna_column(monkeypatch):
    cfg = _bare_instance()
    cfg.fillna = {"missing_col": 0}
    cfg.dummies = {}

    monkeypatch.setattr(
        SurvivalExperimentConfig,
        "clean_data_for_aft",
        staticmethod(lambda data, covariates, target, dummy_dict: data),
    )

    frame = pd.DataFrame({"T": [1], "E": [1]})
    with pytest.raises(ValueError, match="missing_col not found"):
        cfg._prepare_loaded_data(frame)


def test_call_requires_optuna_db_when_optuna_mode(tmp_path):
    cfg = _bare_instance()
    cfg.execution_mode = "optuna"
    cfg.attack_optuna_db = None
    cfg.attack_schema = None
    cfg.attack_query = None
    cfg.plots_folder = str(tmp_path)
    cfg.dummies = None
    cfg.calculate_attack_failures = False

    with pytest.raises(ValueError, match="attack_optuna_db is required"):
        cfg()


def test_call_requires_attack_when_auxiliary_mode(tmp_path):
    cfg = _bare_instance()
    cfg.execution_mode = "auxiliary"
    cfg.attack_optuna_db = None
    cfg.attack = None
    cfg.plots_folder = str(tmp_path)
    cfg.dummies = None
    cfg.calculate_attack_failures = False

    with pytest.raises(ValueError, match="attack is required"):
        cfg()


def test_candidate_attack_metrics_specific_kinds():
    assert (
        SurvivalExperimentConfig._infer_attack_kind_from_label("membership attack")
        == "membership"
    )
    assert (
        SurvivalExperimentConfig._infer_attack_kind_from_label("attribute inference")
        == "attribute"
    )
    assert SurvivalExperimentConfig._infer_attack_kind_from_label("pgd") == "evasion"

    assert SurvivalExperimentConfig._candidate_attack_metrics_for_kind(
        "membership",
    ) == [
        "membership_inference_accuracy",
    ]
    assert SurvivalExperimentConfig._candidate_attack_metrics_for_kind(
        "attribute",
    ) == [
        "sex_inference_accuracy",
        "attribute_inference_accuracy",
    ]


def test_normalize_data_spec_string_and_mapping_for_lifelines():
    string_spec, string_name = SurvivalExperimentConfig._normalize_data_spec(
        data_spec="diabetes",
        target="target",
    )
    assert string_spec["dataset_name"] == "lifelines.diabetes"
    assert string_spec["target"] is None
    assert string_name == "lifelines"

    mapping_spec, mapping_name = SurvivalExperimentConfig._normalize_data_spec(
        data_spec={"dataset_name": "lung", "target": "y"},
        target="target",
    )
    assert mapping_spec["dataset_name"] == "lifelines.lung"
    assert mapping_spec["target"] is None
    assert mapping_name == "lifelines.lung"


def test_load_optuna_frame_query_and_empty(monkeypatch):
    from deckard.layers import compile_results

    monkeypatch.setattr(
        compile_results,
        "parse_studies",
        lambda optuna_db, schema: pd.DataFrame({"score": [0.1, 0.9]}),
    )

    out = SurvivalExperimentConfig._load_optuna_frame(
        optuna_db="sqlite:///optuna.db",
        schema={"sep": "_"},
        query="score > 0.5",
    )
    assert len(out) == 1

    monkeypatch.setattr(
        compile_results,
        "parse_studies",
        lambda optuna_db, schema: pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="No attack results found"):
        SurvivalExperimentConfig._load_optuna_frame(
            optuna_db="sqlite:///optuna.db",
            schema={"sep": "_"},
            query=None,
        )


def test_call_raises_when_aux_runtime_split_missing(monkeypatch, tmp_path):
    cfg = _bare_instance()
    cfg.execution_mode = "auxiliary"
    cfg.attack_optuna_db = None
    cfg.attack_schema = None
    cfg.attack_query = None
    cfg.attack = SimpleNamespace(attack_kind="evasion", attack_size=1)
    cfg.aux_model = lambda runtime_data: {"ok": True}
    cfg.plots_folder = str(tmp_path)
    cfg.dummies = None
    cfg.calculate_attack_failures = False
    cfg.dataset = None
    cfg.test_size = 0.2
    cfg.model_config = {"cox": {"t0": 0.5}}

    monkeypatch.setattr(
        SurvivalExperimentConfig,
        "run_auxiliary_mode",
        classmethod(
            lambda cls, *, data_cfg, survival_config: (
                pd.DataFrame({"T": [1], "E": [1]}),
                survival_config.attack,
                survival_config.aux_model,
            ),
        ),
    )
    monkeypatch.setattr(
        SurvivalExperimentConfig,
        "_prepare_loaded_data",
        lambda self, loaded_data: loaded_data,
    )

    class FakePlotList:
        def __call__(self, **kwargs):
            return {
                "table": pd.DataFrame(),
                "models": {},
                "runtime_data": SimpleNamespace(
                    X_train=None,
                    X_test=None,
                    y_train=None,
                    y_test=None,
                ),
            }

    from deckard.plugins.lifelines import plot as plot_survival_mod

    monkeypatch.setattr(
        plot_survival_mod,
        "SurvivalSeabornPlotConfigList",
        FakePlotList,
    )

    with pytest.raises(ValueError, match="Runtime survival split unavailable"):
        cfg()
