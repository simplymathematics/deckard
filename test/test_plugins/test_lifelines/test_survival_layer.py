import pandas as pd
import pytest

from deckard.data import DataConfig
from deckard.layers.survival import survival_main


def _empty_dataconfig() -> DataConfig:
    return object.__new__(DataConfig)


def _fake_experiment_config(*, data=None, model="weibull", call_fn=None):
    attrs = {
        "data": data if data is not None else _empty_dataconfig(),
        "model": model,
        "target": "E",
        "duration_col": "T",
        "attack": None,
    }
    if call_fn is not None:
        attrs["__call__"] = call_fn
    return type("FakeExperimentConfig", (), attrs)()


def test_survival_main_routes_to_experiment_when_no_plot_payload(monkeypatch):
    from deckard.layers import survival as layer_survival

    calls = {}

    def _fake_call(self):
        calls["data"] = self.data
        calls["model"] = self.model
        return {"aft_table": pd.DataFrame(), "model_scores": None, "models": {}}

    monkeypatch.setattr(
        layer_survival.SurvivalExperimentConfig,
        "__call__",
        _fake_call,
    )

    fake_experiment = _fake_experiment_config(call_fn=_fake_call)

    def _fake_instantiate_config(config_obj, expected_type):
        if expected_type is layer_survival.SurvivalExperimentConfig:
            return fake_experiment
        raise AssertionError(f"Unexpected expected_type: {expected_type}")

    monkeypatch.setattr(layer_survival, "instantiate_config", _fake_instantiate_config)

    cfg = {
        "survival": {
            "data": "lifelines-lung",
            "model": "weibull",
            "target": "E",
            "duration_col": "T",
            "event_col": "E",
        },
    }
    result = survival_main(cfg=cfg)

    assert "aft_table" in result
    assert isinstance(calls["data"], DataConfig)
    assert calls["model"] == "weibull"


def test_survival_main_routes_to_plot_list_for_dataframe_with_plot_payload(
    monkeypatch,
):
    from deckard.layers import survival as layer_survival

    calls = {}

    class FakePlotList:
        def __call__(self, **kwargs):
            calls.update(kwargs)
            return {
                "models": {"weibull": object()},
                "plots": {"weibull": []},
                "table": pd.DataFrame(),
            }

    def _fake_instantiate_config(config_obj, expected_type):
        if expected_type is layer_survival.SurvivalExperimentConfig:
            return _fake_experiment_config()
        if expected_type is layer_survival.SurvivalSeabornPlotConfigList:
            return FakePlotList()
        raise AssertionError(f"Unexpected expected_type: {expected_type}")

    monkeypatch.setattr(layer_survival, "instantiate_config", _fake_instantiate_config)
    monkeypatch.setattr(layer_survival, "_load_plot_dataframe", lambda _: frame)

    frame = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0], "feature": [0.1, 0.2]})
    model_cfg = {"weibull": {"plot": {"summary_file": "s.csv"}, "model": {}}}

    cfg = {
        "survival": {
            "data": "lifelines-lung",
            "model": "weibull",
            "model_config": model_cfg,
            "duration_col": "T",
            "target": "E",
            "event_col": "E",
            "plots_folder": "plots/survival",
        },
    }
    result = survival_main(cfg=cfg)

    assert "models" in result
    assert calls["data"].equals(frame)
    assert calls["model_config"] == model_cfg


def test_survival_main_requires_cfg():
    with pytest.raises(ValueError, match="cfg"):
        survival_main()


@pytest.mark.parametrize(
    ("survival_cfg", "error_type", "error_pattern"),
    [
        ({"data": "   ", "model": "weibull"}, ValueError, "non-empty dataset"),
        ({"data": "lifelines.lung", "model": "   "}, ValueError, "non-empty survival model"),
        ({"data": "lifelines.lung", "model": None}, TypeError, "string or model config mapping"),
    ],
)
def test_validate_raw_data_model_specs_invalid_inputs(
    survival_cfg,
    error_type,
    error_pattern,
):
    from deckard.layers import survival as layer_survival

    with pytest.raises(error_type, match=error_pattern):
        layer_survival._validate_raw_data_model_specs(survival_cfg)


def test_coerce_survival_model_spec_mapping_and_placeholders():
    from deckard.layers import survival as layer_survival

    out = layer_survival._coerce_survival_model_spec(
        {
            "data": "lifelines-lung",
            "model": {"model_type": "lifelines.CoxPHFitter"},
            "plot": {"title": "${model.alias} calibration"},
        },
    )

    assert out["model"] == "cox"
    assert out["plot"]["title"] == "cox calibration"


@pytest.mark.parametrize(
    ("survival_cfg", "expected"),
    [
        ({"plot_only": True}, True),
        ({"model_config": {"weibull": {"plot": {"summary_file": "s.csv"}}}}, True),
        ({"model_config": {"weibull": {"model": {}}}}, False),
        ({}, False),
    ],
)
def test_has_plot_specification_variants(survival_cfg, expected):
    from deckard.layers import survival as layer_survival

    assert layer_survival._has_plot_specification(survival_cfg) is expected


def test_load_plot_dataframe_adds_target_from_y_series():
    from deckard.layers import survival as layer_survival

    data_cfg = _empty_dataconfig()
    data_cfg.X = pd.Series([10.0, 20.0], name="T")
    data_cfg.y = pd.Series([1, 0], name="E")

    experiment_cfg = _fake_experiment_config(data=data_cfg, model="weibull")
    frame = layer_survival._load_plot_dataframe(experiment_cfg)

    assert list(frame.columns) == ["T", "E"]
    assert frame["E"].tolist() == [1, 0]


def test_load_plot_dataframe_validates_required_columns():
    from deckard.layers import survival as layer_survival

    data_cfg = _empty_dataconfig()
    data_cfg.X = pd.DataFrame({"feature": [0.1, 0.2]})
    data_cfg.y = None
    experiment_cfg = _fake_experiment_config(data=data_cfg, model="weibull")

    with pytest.raises(ValueError, match="duration_col"):
        layer_survival._load_plot_dataframe(experiment_cfg)
