from types import ModuleType, SimpleNamespace
import sys

import pytest
from omegaconf import OmegaConf

from deckard.layers import plot as plot_module


def _install_fake_plot_modules(monkeypatch):
    yellow_mod = ModuleType("deckard.plugins.yellowbrick.plot")
    seaborn_mod = ModuleType("deckard.plugins.seaborn.plot")

    class FakeYellowbrickPlotConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self):
            return {"yb": "single"}

    class FakeYellowbrickConfigList:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self):
            return {"yb": "multi"}

    class FakeSeabornPlotConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self):
            return None

    class FakeSeabornPlotConfigList:
        def __init__(self, plots, data_file):
            self.plots = plots
            self.data_file = data_file
            self.file = None

        def __call__(self):
            return None

    yellow_mod.YellowbrickPlotConfig = FakeYellowbrickPlotConfig
    yellow_mod.YellowbrickConfigList = FakeYellowbrickConfigList
    seaborn_mod.SeabornPlotConfig = FakeSeabornPlotConfig
    seaborn_mod.SeabornPlotConfigList = FakeSeabornPlotConfigList

    monkeypatch.setitem(sys.modules, "deckard.plugins.yellowbrick.plot", yellow_mod)
    monkeypatch.setitem(sys.modules, "deckard.plugins.seaborn.plot", seaborn_mod)


def test_load_experiment_config_and_yaml_helpers(tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("model:\n  name: logistic\n")

    loaded = plot_module._load_experiment_config(str(cfg))
    assert loaded["model"]["name"] == "logistic"

    yml = tmp_path / "params.yaml"
    yml.write_text("a: 1\n")
    assert plot_module._load_yaml(str(yml)) == {"a": 1}


def test_load_experiment_config_validates_inputs(tmp_path):
    with pytest.raises(AssertionError, match="Experiment config file not found"):
        plot_module._load_experiment_config(str(tmp_path / "missing.yaml"))

    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(TypeError, match="must resolve to a dictionary"):
        plot_module._load_experiment_config(str(bad))


def test_parse_and_normalize_plots_helpers():
    assert plot_module._parse_plots_arg("a,b, c") == ["a", "b", "c"]
    assert plot_module._parse_plots_arg(["a", "", " b "]) == ["a", "b"]
    assert plot_module._normalize_yellowbrick_plots("all") == "all"
    assert plot_module._normalize_yellowbrick_plots([" all "]) == "all"
    assert plot_module._normalize_yellowbrick_plots(["roc", "pr"]) == ["roc", "pr"]
    assert plot_module._normalize_yellowbrick_plots(123) == ["123"]


def test_instantiate_experiment_cfg_sets_default_target(monkeypatch):
    captured = {}

    def _fake_instantiate(cfg):
        captured.update(cfg)
        return cfg

    monkeypatch.setattr(plot_module, "instantiate", _fake_instantiate)
    out = plot_module._instantiate_experiment_cfg({"data": {}})

    assert out["_target_"] == "deckard.ExperimentConfig"
    assert captured["_target_"] == "deckard.ExperimentConfig"


def test_cfg_resolution_helpers_cover_shapes():
    cfg = OmegaConf.create(
        {
            "plot": {"plot_type": "pairplot", "title": "T"},
            "plot_type": "should_not_win",
            "data_file": "scores.csv",
        },
    )
    resolved = plot_module._resolve_plot_args_from_cfg(cfg)
    assert resolved["plot_type"] == "pairplot"
    assert resolved["title"] == "T"
    assert plot_module._resolve_data_file(cfg) == "scores.csv"

    cfg_top = {"data": {}, "model": {}, "backend": "auto"}
    assert plot_module._extract_experiment_cfg_from_hydra_cfg(cfg_top) == cfg_top

    cfg_nested = {"experiment": {"data": {"x": 1}}}
    assert plot_module._extract_experiment_cfg_from_hydra_cfg(cfg_nested) == {
        "data": {"x": 1},
    }

    cfg_plot_exp = {"plot": {"experiment": {"data": {"y": 2}}}}
    assert plot_module._extract_experiment_cfg_from_hydra_cfg(cfg_plot_exp) == {
        "data": {"y": 2},
    }

    assert (
        plot_module._resolve_experiment_config_path(
            {"plot": {"experiment_config": "a.yaml"}},
        )
        == "a.yaml"
    )
    assert (
        plot_module._resolve_experiment_config_path({"experiment_config": "top.yaml"})
        == "top.yaml"
    )
    assert (
        plot_module._resolve_data_file({"compile_results": {"output_file": "out.csv"}})
        == "out.csv"
    )
    assert plot_module._cfg_to_dict(None) == {}
    assert plot_module._cfg_to_dict(object()) == {}
    assert plot_module._cfg_to_dict([1, 2]) == {}
    assert plot_module._extract_experiment_cfg_from_hydra_cfg({}) == {}


def test_extract_backend_modes_and_validation():
    assert (
        plot_module._extract_backend(
            {"plot": {"backend": "auto"}},
            data_file="scores.csv",
            experiment_cfg={},
            experiment_config="",
        )
        == "seaborn"
    )
    assert (
        plot_module._extract_backend(
            {"backend": "auto"},
            data_file="scores.csv",
            experiment_cfg={"data": {}},
            experiment_config="",
        )
        == "yellowbrick"
    )
    assert (
        plot_module._extract_backend(
            {"backend": "auto"},
            data_file="scores.csv",
            experiment_cfg={},
            experiment_config="exp.yaml",
        )
        == "yellowbrick"
    )
    with pytest.raises(ValueError, match="backend must be one of"):
        plot_module._extract_backend({"backend": "invalid"}, "", {}, "")
    with pytest.raises(ValueError, match="Could not infer backend"):
        plot_module._extract_backend({"backend": "auto"}, "", {}, "")


def test_plot_main_yellowbrick_single_and_multi(monkeypatch, tmp_path):
    _install_fake_plot_modules(monkeypatch)
    monkeypatch.setattr(
        plot_module,
        "_instantiate_experiment_cfg",
        lambda exp_cfg: SimpleNamespace(cfg=exp_cfg),
    )

    single_cfg = {
        "plot": {
            "backend": "yellowbrick",
            "plot_type": "classification_report",
            "plot_folder": str(tmp_path / "plots"),
            "experiment": {"data": {}, "model": {}},
        },
    }
    single = plot_module.plot_main(single_cfg)
    assert single["backend"] == "yellowbrick"
    assert single["mode"] == "single"
    assert single["plot_type"] == "classification_report"

    multi_cfg = {
        "plot": {
            "backend": "yellowbrick",
            "plots": "all",
            "plot_folder": str(tmp_path / "plots_multi"),
            "experiment": {"data": {}, "model": {}},
        },
    }
    multi = plot_module.plot_main(multi_cfg)
    assert multi["backend"] == "yellowbrick"
    assert multi["mode"] == "multi"


def test_plot_main_seaborn_single_and_multi(monkeypatch, tmp_path):
    _install_fake_plot_modules(monkeypatch)

    kwargs_file = tmp_path / "kwargs.yaml"
    rc_file = tmp_path / "rc.yaml"
    plot_params_file = tmp_path / "plots.yaml"
    kwargs_file.write_text("alpha: 0.5\n")
    rc_file.write_text("font.size: 10\n")
    plot_params_file.write_text(
        "plots:\n" "  - plot_type: scatterplot\n" "    x: a\n" "    y: b\n",
    )

    single_cfg = {
        "plot": {
            "backend": "seaborn",
            "data_file": "scores.csv",
            "plot_type": "scatterplot",
            "x": "a",
            "y": "b",
            "kwargs_file": str(kwargs_file),
            "rc_config_file": str(rc_file),
        },
    }
    single = plot_module.plot_main(single_cfg)
    assert single["backend"] == "seaborn"
    assert single["mode"] == "single"

    multi_cfg = {
        "plot": {
            "backend": "seaborn",
            "data_file": "scores.csv",
            "plot_params_file": str(plot_params_file),
            "plot_file": str(tmp_path / "combined.png"),
        },
    }
    multi = plot_module.plot_main(multi_cfg)
    assert multi["backend"] == "seaborn"
    assert multi["mode"] == "multi"
    assert multi["num_plots"] == 1


def test_plot_main_experiment_presence_prefers_yellowbrick(monkeypatch, tmp_path):
    _install_fake_plot_modules(monkeypatch)
    monkeypatch.setattr(
        plot_module,
        "_instantiate_experiment_cfg",
        lambda exp_cfg: SimpleNamespace(cfg=exp_cfg),
    )

    out = plot_module.plot_main(
        {
            "plot": {
                "backend": "auto",
                "data_file": "scores.csv",
                "plot_type": "classification_report",
                "plot_folder": str(tmp_path / "plots"),
                "experiment": {"data": {}, "model": {}},
            },
        },
    )

    assert out["backend"] == "yellowbrick"
    assert out["mode"] == "single"


def test_plot_main_plot_params_file_drives_both_backends(monkeypatch, tmp_path):
    _install_fake_plot_modules(monkeypatch)
    monkeypatch.setattr(
        plot_module,
        "_instantiate_experiment_cfg",
        lambda exp_cfg: SimpleNamespace(cfg=exp_cfg),
    )

    yb_params = tmp_path / "yb_params.yaml"
    yb_params.write_text("plot_params:\n  alpha: 0.5\n")

    seaborn_params = tmp_path / "sns_params.yaml"
    seaborn_params.write_text(
        "plots:\n  - plot_type: scatterplot\n    x: a\n    y: b\n",
    )

    yb = plot_module.plot_main(
        {
            "plot": {
                "backend": "auto",
                "experiment": {"data": {}, "model": {}},
                "plot_type": "classification_report",
                "plot_params_file": str(yb_params),
            },
        },
    )
    assert yb["backend"] == "yellowbrick"

    sns = plot_module.plot_main(
        {
            "plot": {
                "backend": "seaborn",
                "data_file": "scores.csv",
                "plot_params_file": str(seaborn_params),
                "plot_file": str(tmp_path / "combined.png"),
            },
        },
    )
    assert sns["backend"] == "seaborn"
    assert sns["mode"] == "multi"


def test_plot_main_validation_errors(monkeypatch, tmp_path):
    _install_fake_plot_modules(monkeypatch)
    monkeypatch.setattr(
        plot_module,
        "_instantiate_experiment_cfg",
        lambda exp_cfg: SimpleNamespace(cfg=exp_cfg),
    )

    with pytest.raises(ValueError, match="requires plot.experiment_config"):
        plot_module.plot_main({"plot": {"backend": "yellowbrick", "plot_type": "roc"}})

    with pytest.raises(ValueError, match="requires plot.data_file"):
        plot_module.plot_main(
            {"plot": {"backend": "seaborn", "plot_type": "lineplot"}},
        )

    with pytest.raises(ValueError, match="Provide only one"):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "seaborn",
                    "data_file": "scores.csv",
                    "plot_type": "lineplot",
                    "plots": "lineplot",
                },
            },
        )

    with pytest.raises(ValueError, match="requires plot.x and plot.y"):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "seaborn",
                    "data_file": "scores.csv",
                    "plot_type": "lineplot",
                },
            },
        )

    with pytest.raises(
        ValueError,
        match="Provide one of plot.plot_type or plot.plot_params_file for seaborn backend",
    ):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "seaborn",
                    "data_file": "scores.csv",
                    "plots": "lineplot",
                },
            },
        )

    with pytest.raises(ValueError, match="Provide one of plot.plot_type"):
        plot_module.plot_main(
            {"plot": {"backend": "seaborn", "data_file": "scores.csv"}},
        )

    yb_plot_params = tmp_path / "yb_plot_params.yaml"
    yb_plot_params.write_text("plot_params:\n  alpha: 0.5\n")
    yb_out = plot_module.plot_main(
        {
            "plot": {
                "backend": "yellowbrick",
                "plot_type": "classification_report",
                "plot_params_file": str(yb_plot_params),
                "experiment": {"data": {}, "model": {}},
            },
        },
    )
    assert yb_out["backend"] == "yellowbrick"
    assert yb_out["mode"] == "single"

    with pytest.raises(ValueError, match="must contain at least one plot type"):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "yellowbrick",
                    "plots": " , ",
                    "experiment": {"data": {}, "model": {}},
                },
            },
        )

    bad_plot_params = tmp_path / "bad_plot_params.yaml"
    bad_plot_params.write_text("- 1\n- 2\n")
    with pytest.raises(TypeError, match="plot_params_file must contain a dictionary"):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "yellowbrick",
                    "plot_type": "classification_report",
                    "plot_params_file": str(bad_plot_params),
                    "experiment": {"data": {}, "model": {}},
                },
            },
        )

    invalid_kwargs = tmp_path / "bad_kwargs.yaml"
    invalid_kwargs.write_text("- 1\n- 2\n")
    with pytest.raises(TypeError, match="kwargs_file must contain a dictionary"):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "seaborn",
                    "data_file": "scores.csv",
                    "plot_type": "lineplot",
                    "x": "a",
                    "y": "b",
                    "kwargs_file": str(invalid_kwargs),
                },
            },
        )

    bad_plots = tmp_path / "bad_plots.yaml"
    bad_plots.write_text("plots:\n  - 1\n")
    with pytest.raises(TypeError, match="must be a dictionary"):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "seaborn",
                    "data_file": "scores.csv",
                    "plot_params_file": str(bad_plots),
                },
            },
        )

    invalid_rc = tmp_path / "bad_rc.yaml"
    invalid_rc.write_text("- 1\n")
    with pytest.raises(TypeError, match="rc_config_file must contain a dictionary"):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "seaborn",
                    "data_file": "scores.csv",
                    "plot_type": "lineplot",
                    "x": "a",
                    "y": "b",
                    "rc_config_file": str(invalid_rc),
                },
            },
        )

    bad_format = tmp_path / "bad_format.yaml"
    bad_format.write_text("value: 1\n")
    with pytest.raises(
        TypeError,
        match="must contain a list or a dict with key 'plots'",
    ):
        plot_module.plot_main(
            {
                "plot": {
                    "backend": "seaborn",
                    "data_file": "scores.csv",
                    "plot_params_file": str(bad_format),
                },
            },
        )


def test_plot_main_seaborn_multi_accepts_top_level_list_and_merges_defaults(
    monkeypatch,
    tmp_path,
):
    _install_fake_plot_modules(monkeypatch)

    kwargs_file = tmp_path / "kwargs.yaml"
    rc_file = tmp_path / "rc.yaml"
    plot_params_file = tmp_path / "plots_list.yaml"
    kwargs_file.write_text("alpha: 0.5\n")
    rc_file.write_text("font.size: 10\n")
    plot_params_file.write_text("- plot_type: scatterplot\n  x: a\n  y: b\n")

    out = plot_module.plot_main(
        {
            "plot": {
                "backend": "seaborn",
                "data_file": "scores.csv",
                "plot_params_file": str(plot_params_file),
                "kwargs_file": str(kwargs_file),
                "rc_config_file": str(rc_file),
            },
        },
    )

    assert out["backend"] == "seaborn"
    assert out["mode"] == "multi"
    assert out["num_plots"] == 1
