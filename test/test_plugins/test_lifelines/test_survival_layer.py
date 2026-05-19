import pandas as pd

from deckard.data import DataConfig
from deckard.layers.survival import survival_main
from deckard.plugins.lifelines.experiment import LIFELINES_DATASETS, SURVIVAL_MODELS


def _empty_dataconfig() -> DataConfig:
    return object.__new__(DataConfig)


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

    fake_experiment = type(
        "FakeExperimentConfig",
        (),
        {
            "data": _empty_dataconfig(),
            "model": "weibull",
            "target": "E",
            "duration_col": "T",
            "attack": None,
            "__call__": _fake_call,
        },
    )()

    def _fake_instantiate_config(config_obj, expected_type):
        if expected_type is layer_survival.SurvivalExperimentConfig:
            return fake_experiment
        raise AssertionError(f"Unexpected expected_type: {expected_type}")

    monkeypatch.setattr(layer_survival, "instantiate_config", _fake_instantiate_config)

    cfg = {
        "survival": {
            "data": dict(LIFELINES_DATASETS["lung"]),
            "model": SURVIVAL_MODELS["weibull"],
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
            return type(
                "FakeExperimentConfig",
                (),
                {
                    "data": _empty_dataconfig(),
                    "model": "weibull",
                    "target": "E",
                    "duration_col": "T",
                    "attack": None,
                },
            )()
        if expected_type is layer_survival.SurvivalSeabornPlotConfigList:
            return FakePlotList()
        raise AssertionError(f"Unexpected expected_type: {expected_type}")

    monkeypatch.setattr(layer_survival, "instantiate_config", _fake_instantiate_config)
    monkeypatch.setattr(layer_survival, "_load_plot_dataframe", lambda _: frame)

    frame = pd.DataFrame({"T": [1.0, 2.0], "E": [1, 0], "feature": [0.1, 0.2]})
    model_cfg = {"weibull": {"plot": {"summary_file": "s.csv"}, "model": {}}}

    cfg = {
        "survival": {
            "data": dict(LIFELINES_DATASETS["lung"]),
            "model": SURVIVAL_MODELS["weibull"],
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
    try:
        survival_main()
    except ValueError as exc:
        assert "cfg" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError when cfg is not provided")


def test_survival_main_rejects_non_object_data_and_model():
    cfg = {
        "survival": {
            "data": "lifelines.lung",
            "model": "weibull",
        },
    }

    try:
        survival_main(cfg=cfg)
    except TypeError as exc:
        assert "DataConfig" in str(exc)
    else:
        raise AssertionError("Expected TypeError for non-object data/model inputs")
