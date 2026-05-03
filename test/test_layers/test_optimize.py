import json
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf
import torch
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

    def _get_file_dict(self):
        return self.to_dict()


class DummyConf:
    def __init__(self):
        self.files = DummyFiles()
        self.experiment_name = None
        self.post_init_calls = 0

    def __post_init__(self):
        self.post_init_calls += 1


class DummyStorage:
    def __init__(self):
        self.attrs = {}

    def set_trial_user_attr(self, trial_id, key, value):
        self.attrs[(trial_id, key)] = value


class DummyTrial:
    def __init__(self, number, trial_id, user_attrs=None):
        self.number = number
        self._trial_id = trial_id
        self.user_attrs = user_attrs or {}


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


def test_filter_scores_marks_missing_optimizer_score_as_fallback_value():
    scores = {"loss": 0.2, "latency": 12.0}

    values, attrs = optimize_module.filter_scores(
        scores,
        ["loss", "accuracy"],
        ["minimize", "diff"],
    )

    assert values == 0.2
    assert attrs["accuracy"] == float("inf")
    assert attrs["latency"] == 12.0


def test_filter_scores_raises_for_invalid_direction():
    with pytest.raises(ValueError, match="Invalid direction"):
        optimize_module.filter_scores(
            {"loss": 0.2},
            ["loss"],
            ["invalid"],
        )


def test_filter_scores_raises_when_no_optimization_values_exist():
    with pytest.raises(RuntimeError, match="No optimization scores found"):
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

    monkeypatch.setattr(
        optimize_module.optuna, "create_study", fake_create_study
    )

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

    monkeypatch.setattr(
        optimize_module.optuna, "create_study", fake_create_study
    )

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


def test_create_study_filters_diff_direction_for_optuna(monkeypatch):
    calls = {}

    def fake_create_study(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(
        optimize_module.optuna, "create_study", fake_create_study
    )

    optimize_module.create_study(
        "study",
        "sqlite:///db.sqlite3",
        ["minimize", "diff", "maximize"],
        ["loss", "latency_delta", "accuracy"],
    )

    assert calls == {
        "study_name": "study",
        "storage": "sqlite:///db.sqlite3",
        "directions": ["minimize", "maximize"],
        "load_if_exists": True,
    }


def test_create_study_allows_only_diff_directions(monkeypatch):
    calls = {}

    def fake_create_study(**kwargs):
        calls.update(kwargs)
        return object()

    monkeypatch.setattr(
        optimize_module.optuna, "create_study", fake_create_study
    )

    optimize_module.create_study(
        "study",
        "sqlite:///db.sqlite3",
        ["diff"],
        ["latency_delta"],
    )

    assert calls == {
        "study_name": "study",
        "storage": "sqlite:///db.sqlite3",
        "load_if_exists": True,
    }


def test_create_study_requires_matching_directions_and_optimizers():
    with pytest.raises(
        AssertionError,
        match="Length of directions must match length of optimizers",
    ):
        optimize_module.create_study(
            "study", "sqlite:///db.sqlite3", ["minimize"], []
        )


def test_hydra_optuna_callback_sets_up_study(monkeypatch):
    study = DummyStudy()
    captured = {}

    def fake_create_study(study_name, storage, directions, optimizers):
        captured["study_name"] = study_name
        captured["storage"] = storage
        captured["directions"] = directions
        captured["optimizers"] = optimizers
        return study

    monkeypatch.setattr(optimize_module, "create_study", fake_create_study)

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///db.sqlite3",
        directions=["minimize", "diff", "maximize"],
        optimizers=["loss", "delta", "accuracy"],
    )

    callback.on_multirun_start(OmegaConf.create({}))

    assert captured == {
        "study_name": "demo-study",
        "storage": "sqlite:///db.sqlite3",
        "directions": ["minimize", "diff", "maximize"],
        "optimizers": ["loss", "delta", "accuracy"],
    }
    assert study.metric_names == ["loss", "accuracy"]


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


def test_set_study_metric_names_filters_diff_direction():
    study = DummyStudy()

    optimize_module.set_study_metric_names(
        study,
        ["loss", "latency_delta", "accuracy"],
        ["minimize", "diff", "maximize"],
    )

    assert study.metric_names == ["loss", "accuracy"]


def test_set_user_attrs_accepts_dictconfig():
    study = DummyStudy()
    attrs = OmegaConf.create({"fold": 1, "tag": "baseline"})

    optimize_module.set_study_attributes(study, attrs)

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
    conf.experiment_name = "security_classification_linear_hsj"
    hydra_cfg = SimpleNamespace(
        job=SimpleNamespace(num=7, name="optimize"),
        sweep=SimpleNamespace(dir=str(tmp_path), subdir="run_7"),
    )

    result = optimize_module.prepare_multirun_file_paths(hydra_cfg, conf)

    assert result is conf
    assert conf.experiment_name == optimize_module.hash_conf_values(
        "security_classification_linear_hsj",
    )
    assert conf.post_init_calls == 1
    assert conf.files.log_file == str(tmp_path / "run_7" / "optimize.log")
    assert conf.files.score_file == str(tmp_path / "run_7" / "scores.json")
    assert conf.files.params_file == str(tmp_path / "run_7" / "params.yaml")
    assert conf.files.error_file == str(tmp_path / "run_7" / "error.log")
    assert conf.files.post_init_calls == 1


def test_optimize_multirun_relies_on_callback_for_params_and_scores(
    monkeypatch,
    tmp_path,
):
    class MultirunConf:
        def __init__(self):
            self.files = DummyFiles(tmp_path)
            self.optimizers = ["loss"]
            self.directions = ["minimize"]
            self.experiment_name = "security_classification_linear_hsj"

        def execute_without_mercy(self):
            return {"loss": 0.25, "accuracy": 0.9}

    conf = MultirunConf()
    hydra_cfg = OmegaConf.create(
        {
            "sweeper": {
                "storage": "sqlite:///study.sqlite3",
                "study_name": "demo-study",
            },
            "job": {"id": 0},
        },
    )

    monkeypatch.setattr(
        optimize_module,
        "prepare_multirun_file_paths",
        lambda hydra_cfg, conf_obj: conf_obj,
    )

    result = optimize_module.optimize_multirun(
        {"foo": "bar"},
        hydra_cfg,
        conf,
    )

    assert result == 0.25
    assert not (tmp_path / "scores.json").exists()
    assert not (tmp_path / "params.yaml").exists()


def test_hydra_optuna_callback_on_compose_config_sets_experiment_and_files(
    monkeypatch,
    tmp_path,
):
    hydra_cfg = SimpleNamespace(
        mode="RunMode.MULTIRUN",
        sweeper={
            "storage": "sqlite:///study.sqlite3",
            "study_name": "demo-study",
        },
        sweep=SimpleNamespace(dir=str(tmp_path), subdir="run_3"),
        job=SimpleNamespace(name="optimize"),
    )
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
    )
    cfg = OmegaConf.create({"name": "demo", "files": {}})

    callback.on_compose_config(cfg)

    assert isinstance(cfg.experiment_name, str)
    assert len(cfg.experiment_name) == 32
    assert cfg.files.log_file == str(tmp_path / "run_3" / "optimize.log")
    assert cfg.files.score_file == str(tmp_path / "run_3" / "scores.json")
    assert cfg.files.params_file == str(tmp_path / "run_3" / "params.yaml")
    assert cfg.files.error_file == str(tmp_path / "run_3" / "error.log")


def test_hydra_optuna_callback_on_job_start_writes_params_file(
    monkeypatch, tmp_path
):
    hydra_cfg = SimpleNamespace(mode="RunMode.MULTIRUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
    )
    cfg = OmegaConf.create(
        {
            "name": "demo",
            "files": {"params_file": str(tmp_path / "params.yaml")},
        },
    )

    callback.on_job_start(cfg)

    assert (tmp_path / "params.yaml").exists()
    assert "name: demo" in (tmp_path / "params.yaml").read_text()


def test_hydra_optuna_callback_on_job_end_writes_score_file(
    monkeypatch, tmp_path
):
    hydra_cfg = SimpleNamespace(mode="RunMode.MULTIRUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
    )
    cfg = OmegaConf.create(
        {
            "name": "demo",
            "files": {"score_file": str(tmp_path / "scores.json")},
        },
    )

    callback.on_job_end(
        cfg,
        job_return=SimpleNamespace(
            return_value={"loss": 0.25, "accuracy": 0.9}
        ),
    )

    assert json.loads((tmp_path / "scores.json").read_text()) == {
        "loss": 0.25,
        "accuracy": 0.9,
    }


def test_set_trial_attributes_persists_all_attrs_via_storage():
    storage = DummyStorage()
    experiment_name = "security_classification_linear_hsj"
    exp_hash = optimize_module.hash_conf_values(experiment_name)
    trial = DummyTrial(
        number=7,
        trial_id=101,
        user_attrs={"experiment_name": exp_hash},
    )
    study = SimpleNamespace(
        study_name="demo-study",
        _storage=storage,
        get_trials=lambda deepcopy=False: [trial],
    )

    attrs = OmegaConf.create(
        {"accuracy": 0.91, "latency_ms": 12.5, "meta": {"fold": 1}},
    )

    optimize_module.set_trial_attributes(
        study,
        attrs,
        experiment_name=experiment_name,
    )

    assert storage.attrs[(101, "accuracy")] == 0.91
    assert storage.attrs[(101, "latency_ms")] == 12.5
    assert storage.attrs[(101, "meta")] == {"fold": 1}
    assert storage.attrs[(101, "experiment_name")] == exp_hash


def test_set_trial_attributes_skips_when_experiment_uuid_missing(caplog):
    experiment_name = "security_classification_linear_hsj"
    storage = DummyStorage()
    study = SimpleNamespace(
        study_name="demo-study",
        _storage=storage,
        get_trials=lambda deepcopy=False: [
            DummyTrial(
                number=1,
                trial_id=11,
                user_attrs={"experiment_name": "different_hash"},
            ),
        ],
    )

    optimize_module.set_trial_attributes(
        study,
        {"accuracy": 0.9},
        experiment_name=experiment_name,
    )

    assert storage.attrs == {}
    assert "Skipping trial attribute sync" in caplog.text


def test_optimize_main_executes_conf_object_in_single_run(monkeypatch):
    class DummyBase:
        def __call__(self):
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

    hydra_cfg = SimpleNamespace(
        mode="RunMode.MULTIRUN",
        sweeper={"storage": "sqlite:///db.sqlite3", "study_name": "demo-study"},
    )

    monkeypatch.setattr(optimize_module, "ConfigBase", DummyBase)
    monkeypatch.setattr(optimize_module, "ExperimentConfig", DummyExperiment)
    monkeypatch.setattr(optimize_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(
        optimize_module, "optimize_multirun", fake_optimize_multirun
    )
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    result = optimize_module.optimize_main(OmegaConf.create({"name": "demo"}))

    assert result == {"best": 0.1}
    assert captured["conf_obj"] is conf_obj
    assert captured["hydra_cfg"] is hydra_cfg
    assert isinstance(captured["multirun_cfg"], str)
    assert "name: demo" in captured["multirun_cfg"]
    assert "experiment_name:" in captured["multirun_cfg"]
    assert captured["cfg"][
        "experiment_name"
    ] == optimize_module.hash_conf_values(
        _root_={"name": "demo"},
    )


def test_optimize_main_runs_hydra_configured_pytorch_experiment(monkeypatch):
    hydra_cfg = SimpleNamespace(mode="RunMode.RUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    X = torch.randn(48, 4)
    y = torch.randint(0, 2, (48,))
    dataset = torch.utils.data.TensorDataset(X, y)

    monkeypatch.setattr(
        "deckard.data.pytorch.load_class",
        lambda *_args, **_kwargs: dataset,
    )

    cfg = OmegaConf.create(
        {
            "_target_": "deckard.experiment.ExperimentConfig",
            "library": "pytorch",
            "classifier": True,
            "data": {
                "_target_": "deckard.data.pytorch.PytorchDataConfig",
                "dataset_name": "torch.utils.data.TensorDataset",
                "data_params": {},
                "train_size": 32,
                "test_size": 16,
                "stratify": True,
            },
            "model": {
                "_target_": "deckard.model.pytorch.PytorchModelConfig",
                "model_type": "torch.nn.Linear",
                "model_params": {"in_features": 4, "out_features": 2},
                "classifier": True,
                "fit_params": {"nb_epochs": 1, "batch_size": 8},
                "criterion": "CrossEntropyLoss",
                "optimizer": {"name": "SGD", "lr": 0.01},
            },
            "attack": None,
            "files": {"_target_": "deckard.file.FileConfig"},
        },
    )

    scores = optimize_module.optimize_main(cfg)

    assert "accuracy" in scores
    assert "optimizer_loss" in scores
