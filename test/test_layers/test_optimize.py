import json
import math
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
from deckard.layers import optimize as optimize_module

class DummyStudy:
    def __init__(self):
        self.metric_names = None
        self.user_attrs = {}

    def set_metric_names(self, names):
        self.metric_names = list(names)

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class DummyFiles:
    def __init__(self, tmp_path=None):
        self.experiment_name = None
        self.post_init_calls = 0
        if tmp_path is not None:
            self.log_file = str(tmp_path / "run.log")
            self.score_file = str(tmp_path / "scores.json")
            self.params_file = str(tmp_path / "params.yaml")
            self.error_file = str(tmp_path / "error.log")

    def __post_init__(self):
        self.post_init_calls += 1

    def to_dict(self):
        return {
            "log_file": self.log_file,
            "score_file": self.score_file,
            "params_file": self.params_file,
            "error_file": self.error_file,
        }


class DummyConf:
    def __init__(self):
        self.files = DummyFiles()
        self.experiment_name = None
        self.post_init_calls = 0

    def __post_init__(self):
        self.post_init_calls += 1


def test_filter_scores_returns_original_scores_when_no_optimizers():
    scores = {"loss": 0.2, "accuracy": 0.9}

    values, attrs = optimize_module.filter_scores(scores, [], [])

    assert values == scores
    assert attrs == {}


def test_filter_scores_splits_optimized_values_and_attributes():
    scores = {"loss": 0.2, "accuracy": 0.9, "latency": 12.0}

    values, attrs = optimize_module.filter_scores(
        scores,
        ["loss", "accuracy"],
        ["minimize", "diff"],
    )

    assert values == 0.2
    assert attrs == {"accuracy": 0.9, "latency": 12.0}


def test_filter_scores_marks_missing_optimizer_score_as_nan():
    scores = {"loss": 0.2, "latency": 12.0}

    values, attrs = optimize_module.filter_scores(
        scores,
        ["loss", "accuracy"],
        ["minimize", "diff"],
    )

    assert values == 0.2
    assert math.isnan(attrs["accuracy"])
    assert attrs["latency"] == 12.0


def test_filter_scores_raises_for_invalid_direction():
    with pytest.raises(ValueError, match="Invalid direction"):
        optimize_module.filter_scores(
            {"loss": 0.2},
            ["loss"],
            ["invalid"],
        )


def test_filter_scores_raises_when_no_optimization_values_exist():
    with pytest.raises(ValueError, match="No optimization scores found"):
        optimize_module.filter_scores(
            {"accuracy": 0.9},
            ["accuracy"],
            ["diff"],
        )


def test_create_study_without_directions(monkeypatch):
    calls = {}

    def fake_create_study(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(optimize_module.optuna, "create_study", fake_create_study)

    optimize_module.create_study("study", "sqlite:///db.sqlite3", [], [])

    assert calls == {
        "study_name": "study",
        "storage": "sqlite:///db.sqlite3",
        "load_if_exists": True,
    }


def test_create_study_with_directions(monkeypatch):
    calls = {}

    def fake_create_study(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(optimize_module.optuna, "create_study", fake_create_study)

    optimize_module.create_study(
        "study",
        "sqlite:///db.sqlite3",
        ["minimize", "maximize"],
        ["loss", "accuracy"],
    )

    assert calls == {
        "study_name": "study",
        "storage": "sqlite:///db.sqlite3",
        "directions": ["minimize", "maximize"],
        "load_if_exists": True,
    }


def test_create_study_requires_matching_directions_and_optimizers():
    with pytest.raises(AssertionError, match="Length of directions must match length of optimizers"):
        optimize_module.create_study("study", "sqlite:///db.sqlite3", ["minimize"], [])


@pytest.mark.parametrize(
    ("optimizers", "expected"),
    [
        ("loss", ["loss"]),
        (("loss", "accuracy"), ["loss", "accuracy"]),
        (OmegaConf.create(["loss", "accuracy"]), ["loss", "accuracy"]),
    ],
)
def test_set_study_metric_names_accepts_supported_types(optimizers, expected):
    study = DummyStudy()

    optimize_module.set_study_metric_names(study, optimizers)

    assert study.metric_names == expected



def test_set_user_attrs_accepts_dictconfig():
    study = DummyStudy()
    attrs = OmegaConf.create({"fold": 1, "tag": "baseline"})

    optimize_module.set_user_attrs(study, attrs)

    assert study.user_attrs == {"fold": 1, "tag": "baseline"}


def test_save_params_file_writes_config_without_params(tmp_path):
    cfg = {"params": {"lr": 0.1}, "trainer": {"epochs": 5}}
    files = {"params_file": str(tmp_path / "params.yaml")}

    result = optimize_module.save_params_file(cfg, files)

    saved = OmegaConf.load(files["params_file"])

    assert "params" not in cfg
    assert "params" not in result
    assert saved.trainer.epochs == 5


def test_save_params_file_requires_params_file():
    with pytest.raises(ValueError, match="params_file must be specified"):
        optimize_module.save_params_file({}, {})


def test_prepare_multirun_file_paths_updates_conf_and_files(tmp_path):
    conf = DummyConf()
    hydra_cfg = SimpleNamespace(
        job=SimpleNamespace(num=7, name="optimize"),
        sweep=SimpleNamespace(dir=str(tmp_path), subdir="run_7"),
    )

    result = optimize_module.prepare_multirun_file_paths(hydra_cfg, conf)

    assert result is conf
    assert conf.experiment_name == "7"
    assert conf.post_init_calls == 1
    assert conf.files.log_file == str(tmp_path / "run_7" / "optimize.log")
    assert conf.files.score_file == str(tmp_path / "run_7" / "scores.json")
    assert conf.files.params_file == str(tmp_path / "run_7" / "params.yaml")
    assert conf.files.error_file == str(tmp_path / "run_7" / "error.log")
    assert conf.files.post_init_calls == 1


def test_optimize_multirun_writes_files_and_updates_study(monkeypatch, tmp_path):
    study = DummyStudy()

    class MultirunConf:
        def __init__(self):
            self.files = DummyFiles(tmp_path)
            self.optimizers = ["loss"]
            self.directions = ["minimize"]

        def execute(self):
            return {"loss": 0.25, "accuracy": 0.9}

    conf = MultirunConf()
    hydra_cfg = OmegaConf.create(
        {
            "sweeper": {
                "storage": "sqlite:///study.sqlite3",
                "study_name": "demo-study",
            }
        }
    )
    captured = {}

    monkeypatch.setattr(
        optimize_module,
        "prepare_multirun_file_paths",
        lambda hydra_cfg, conf_obj: conf_obj,
    )

    def fake_create_study(study_name, storage, directions, optimizers):
        captured["study_name"] = study_name
        captured["storage"] = storage
        captured["directions"] = directions
        captured["optimizers"] = optimizers
        return study

    monkeypatch.setattr(optimize_module, "create_study", fake_create_study)

    result = optimize_module.optimize_multirun(
        OmegaConf.create({"foo": "bar"}),
        hydra_cfg,
        conf,
    )

    assert result == 0.25
    assert captured == {
        "study_name": "demo-study",
        "storage": "sqlite:///study.sqlite3",
        "directions": ["minimize"],
        "optimizers": ["loss"],
    }
    assert study.metric_names == ["loss"]
    assert study.user_attrs == {"accuracy": 0.9}
    assert json.loads((tmp_path / "scores.json").read_text()) == {"loss": 0.25, "accuracy": 0.9}
    assert "foo: bar" in (tmp_path / "params.yaml").read_text()


def test_optimize_main_executes_conf_object_in_single_run(monkeypatch):
    class DummyBase:
        def execute(self):
            return {"score": 1.0}

    captured = {}

    def fake_instantiate(cfg):
        captured["cfg"] = cfg
        return DummyBase()

    monkeypatch.setattr(optimize_module, "ConfigBase", DummyBase)
    monkeypatch.setattr(optimize_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(
        optimize_module.HydraConfig,
        "get",
        lambda: SimpleNamespace(mode="RunMode.RUN"),
    )

    result = optimize_module.optimize_main(OmegaConf.create({"name": "demo"}))

    assert result == {"score": 1.0}
    assert captured["cfg"]["_target_"] == "deckard.ExperimentConfig"


def test_optimize_main_uses_multirun_path(monkeypatch):
    class DummyBase:
        pass

    class DummyExperiment(DummyBase):
        pass

    conf_obj = DummyExperiment()
    captured = {}

    def fake_instantiate(cfg):
        captured["cfg"] = cfg
        return conf_obj

    def fake_optimize_multirun(cfg, hydra_cfg, obj):
        captured["multirun_cfg"] = cfg
        captured["hydra_cfg"] = hydra_cfg
        captured["conf_obj"] = obj
        return {"best": 0.1}

    hydra_cfg = SimpleNamespace(mode="RunMode.MULTIRUN")

    monkeypatch.setattr(optimize_module, "ConfigBase", DummyBase)
    monkeypatch.setattr(optimize_module, "ExperimentConfig", DummyExperiment)
    monkeypatch.setattr(optimize_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(optimize_module, "optimize_multirun", fake_optimize_multirun)
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    result = optimize_module.optimize_main(OmegaConf.create({"name": "demo"}))

    assert result == {"best": 0.1}
    assert captured["conf_obj"] is conf_obj
    assert captured["hydra_cfg"] is hydra_cfg
    assert captured["multirun_cfg"]["_target_"] == "deckard.ExperimentConfig"