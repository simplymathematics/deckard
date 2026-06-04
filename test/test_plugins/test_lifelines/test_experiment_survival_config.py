from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deckard.data import DataConfig
from deckard.experiment.base import ExperimentConfig
from deckard.model import ModelConfig, SurvivalModelConfig
from deckard.plugins.lifelines.experiment import SurvivalExperimentConfig


def test_allows_aux_model_without_attack_config():
    cfg = SurvivalExperimentConfig(
        data=DataConfig(
            name="make_regression",
            classifier=False,
        ),
        model="cox",
        target="E",
        classifier=False,
        aux_model=ModelConfig(
            name="sklearn.linear_model.LogisticRegression",
            classifier=True,
            model_params={"max_iter": 10},
        ),
        duration_col="T",
        event_col="E",
    )
    assert cfg.aux_model is not None


def test_requires_data_config():
    with pytest.raises(ValueError):
        SurvivalExperimentConfig(
            data=None,
            model="cox",
            target="E",
            duration_col="T",
            event_col="E",
            classifier=False,
        )


@pytest.mark.parametrize(
    "data_cfg",
    [
        DataConfig(name="make_regression", classifier=False),
        DataConfig(name="make_regression", classifier=False, target=None),
    ],
)
def test_survival_config_initialization_variants(data_cfg):
    config = SurvivalExperimentConfig(
        data=data_cfg,
        model="cox",
        target="E",
        classifier=False,
        duration_col="T",
        event_col="E",
    )
    assert isinstance(config, SurvivalExperimentConfig)
    assert config.model == "cox"


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
    cfg.model = "weibull"
    cfg.data = "lifelines-diabetes"
    cfg.__post_init__()
    assert isinstance(cfg.data, DataConfig)

    cfg = _bare_instance()
    cfg.model = ModelConfig(
        name="sklearn.tree.DecisionTreeClassifier",
        classifier=True,
        model_params={"max_depth": 1},
    )
    with pytest.raises((ValueError, TypeError)):
        cfg.__post_init__()

    cfg = _bare_instance()
    cfg.model = "weibull"
    cfg.duration_col = ""
    with pytest.raises(ValueError, match="duration_col must be a non-empty string"):
        cfg.__post_init__()


@pytest.mark.parametrize(
    ("label", "expected_kind"),
    [
        (np.nan, None),
        ("   ", None),
        ("membership attack", "membership"),
        ("attribute inference", "attribute"),
        ("pgd", "evasion"),
    ],
)
def test_infer_attack_kind_from_label_variants(label, expected_kind):
    assert (
        SurvivalExperimentConfig._infer_attack_kind_from_label(label) == expected_kind
    )


@pytest.mark.parametrize(
    ("kind", "expected_metrics"),
    [
        (
            None,
            [
                "evasion_success",
                "evasion_accuracy",
                "membership_inference_accuracy",
                "attribute_inference_accuracy",
            ],
        ),
        ("membership", ["membership_inference_accuracy"]),
        ("attribute", ["sex_inference_accuracy", "attribute_inference_accuracy"]),
    ],
)
def test_candidate_attack_metrics_for_kind_variants(kind, expected_metrics):
    metrics = SurvivalExperimentConfig._candidate_attack_metrics_for_kind(kind)
    for metric in expected_metrics:
        assert metric in metrics


def test_resolve_attack_size_from_uniform_column_without_row_index():
    df = pd.DataFrame({"attack_size": [5, 5, 5]})
    size = SurvivalExperimentConfig._resolve_attack_size(df)
    assert size == 5.0


def test_calculate_failures_from_signals_fallback_uses_attack_size_column():
    cfg = _bare_instance()
    df = pd.DataFrame(
        {
            "accuracy": [0.5, 0.5],
            "evasion_accuracy": [0.2, 0.4],
            "attack_size": [10.0, np.nan],
        },
    )

    out = cfg.calculate_failures_from_signals(
        df,
        failure_profile=SimpleNamespace(attack_size=8, attack_kind="evasion"),
    )

    assert "ben_failures" in out.columns
    assert "adv_failures" in out.columns
    assert np.isfinite(out["adv_failures"]).all()


def test_make_survival_model_table_handles_none_models_and_metric_failures(
    monkeypatch,
    tmp_path,
):
    cfg = SurvivalModelConfig(duration_col="T", event_col="E", t0=0.35)

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

    monkeypatch.setattr(
        cfg,
        "survival_probability_calibration",
        lambda **kwargs: (None, 0.1, 0.2),
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
        SurvivalModelConfig,
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

    # Auxiliary mode can now run without attack config for non-attack failures.
    cfg.attack = None
    cfg.aux_model = None
    cfg.dataset = None
    cfg.test_size = 0.2
    cfg.model_config = {"cox": {"t0": 0.5}}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        SurvivalExperimentConfig,
        "run_auxiliary_mode",
        classmethod(
            lambda cls, *, data_cfg, survival_config: (
                pd.DataFrame({"T": [1], "E": [1], "failure_rate": [0.2]}),
                None,
                None,
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
                "runtime_data": SimpleNamespace(),
            }

    from deckard.plugins.lifelines import plot as plot_survival_mod

    monkeypatch.setattr(
        plot_survival_mod,
        "SurvivalSeabornPlotConfigList",
        FakePlotList,
    )

    out = cfg()
    assert "aft_table" in out
    monkeypatch.undo()


def test_calculate_failures_from_non_attack_metrics():
    cfg = _bare_instance()
    cfg.target = "failure_rate"
    frame = pd.DataFrame(
        {
            "failure_rate": [0.1, 0.3, 0.2],
            "T": [1, 2, 3],
            "E": [1, 1, 0],
        },
    )
    out = cfg.calculate_failures_from_signals(frame)
    assert "adv_failures" in out.columns
    assert np.allclose(out["adv_failures"], frame["failure_rate"])


def test_normalize_data_spec_string_and_mapping_for_lifelines():
    string_spec, string_name = SurvivalExperimentConfig._normalize_data_spec(
        data_spec="diabetes",
        target="target",
    )
    assert string_spec["name"] == "lifelines.diabetes"
    assert string_spec["target"] is None
    assert string_name == "lifelines"

    mapping_spec, mapping_name = SurvivalExperimentConfig._normalize_data_spec(
        data_spec={"name": "lung", "target": "y"},
        target="target",
    )
    assert mapping_spec["name"] == "lifelines.lung"
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


def test_build_survival_plot_config_list_model():
    """A list of model names should produce one config entry per model."""
    cfg = SurvivalExperimentConfig(
        data=DataConfig(name="lifelines_rossi"),
        model=["weibull", "cox"],
        target="arrest",
        event_col="arrest",
        duration_col="week",
    )
    result = cfg._build_survival_plot_config()
    assert set(result.keys()) == {"weibull", "cox"}
    assert result["weibull"]["t0"] == cfg.t0
    assert result["cox"]["t0"] == cfg.t0


def test_build_survival_plot_config_dict_model():
    """A dict model spec should be used directly as the plot config."""
    cfg = SurvivalExperimentConfig(
        data=DataConfig(name="lifelines_rossi"),
        model={"weibull": {"t0": 0.5}, "cox": {}},
        target="arrest",
        event_col="arrest",
        duration_col="week",
    )
    result = cfg._build_survival_plot_config()
    assert set(result.keys()) == {"weibull", "cox"}
    assert result["weibull"]["t0"] == 0.5


def test_build_survival_plot_config_string_model():
    """A string model name should produce a single-entry config (backward compat)."""
    cfg = SurvivalExperimentConfig(
        data=DataConfig(name="lifelines_rossi"),
        model="weibull",
        target="arrest",
        event_col="arrest",
        duration_col="week",
    )
    result = cfg._build_survival_plot_config()
    assert list(result.keys()) == ["weibull"]
    assert result["weibull"]["t0"] == cfg.t0


def test_validate_model_list_rejects_empty():
    """An empty model list should raise at construction time."""
    with pytest.raises((ValueError, TypeError)):
        SurvivalExperimentConfig(
            data=DataConfig(name="lifelines_rossi"),
            model=[],
            target="arrest",
            event_col="arrest",
            duration_col="week",
        )
