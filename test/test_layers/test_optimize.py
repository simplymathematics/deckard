import contextlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import optuna
import pytest
import torch
from helpers import make_runtime_env
from omegaconf import OmegaConf

from deckard.layers import optimize as optimize_module
from deckard.experiment.canon import CANONICAL_EXPERIMENT_STAGE_COMPONENTS

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_SKLEARN_DIR = ROOT / "examples" / "sklearn"
DECKARD_RC_PATH = EXAMPLES_SKLEARN_DIR / ".deckard_rc"


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
        optimize_module.optuna,
        "create_study",
        fake_create_study,
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
        optimize_module.optuna,
        "create_study",
        fake_create_study,
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
        optimize_module.optuna,
        "create_study",
        fake_create_study,
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
        optimize_module.optuna,
        "create_study",
        fake_create_study,
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
            "study",
            "sqlite:///db.sqlite3",
            ["minimize"],
            [],
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
    monkeypatch.setattr(
        optimize_module.HydraConfig,
        "get",
        lambda: SimpleNamespace(mode="RunMode.MULTIRUN", sweeper=None),
    )

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


def test_execute_runtime_object_executes_without_mercy_once(tmp_path):
    class RuntimeConf:
        def __init__(self):
            self.files = DummyFiles(tmp_path)
            self.optimizers = ["loss"]
            self.directions = ["minimize"]
            self.experiment_name = "security_classification_linear_hsj"

        def execute_without_mercy(self):
            return {"loss": 0.25, "accuracy": 0.9}

    conf = RuntimeConf()
    result = optimize_module.OptunaStudyCallback.execute_runtime_object(conf)

    assert result == {"loss": 0.25, "accuracy": 0.9}
    assert not (tmp_path / "scores.json").exists()
    assert not (tmp_path / "params.yaml").exists()


def test_execute_runtime_object_rejects_non_mapping_payload(tmp_path):
    class RuntimeConf:
        def __init__(self):
            self.files = DummyFiles(tmp_path)
            self.optimizers = ["loss"]
            self.directions = ["minimize"]
            self.experiment_name = "security_classification_linear_hsj"

        def execute_without_mercy(self):
            return 0.2

    conf = RuntimeConf()
    with pytest.raises(TypeError, match="must return a dict-like score payload"):
        optimize_module.OptunaStudyCallback.execute_runtime_object(conf)


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


def test_hydra_optuna_callback_on_compose_config_writes_params_file(
    monkeypatch,
    tmp_path,
):
    hydra_cfg = SimpleNamespace(mode="RunMode.MULTIRUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)
    monkeypatch.setattr(
        optimize_module,
        "_normalize_mode_cfg",
        lambda cfg, h, **kw: cfg,
    )
    monkeypatch.setattr(
        optimize_module,
        "_seed_experiment_uuid_for_current_trial",
        lambda **kw: None,
    )

    params_file = str(tmp_path / "params.yaml")
    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
        params_file=params_file,
    )
    cfg = OmegaConf.create(
        {
            "name": "demo",
        },
    )

    callback.on_compose_config(cfg)

    assert (tmp_path / "params.yaml").exists()
    assert "name: demo" in (tmp_path / "params.yaml").read_text()


def test_hydra_optuna_callback_on_job_end_writes_score_file(
    monkeypatch,
    tmp_path,
):
    hydra_cfg = SimpleNamespace(mode="RunMode.MULTIRUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
        score_file=str(tmp_path / "scores.json"),
    )
    # Simulate on_compose_config having resolved the score_file
    callback._resolved_score_file = str(tmp_path / "scores.json")
    cfg = OmegaConf.create(
        {
            "name": "demo",
        },
    )

    callback.on_job_end(
        cfg,
        job_return=SimpleNamespace(
            return_value={"loss": 0.25, "accuracy": 0.9},
        ),
    )

    assert json.loads((tmp_path / "scores.json").read_text()) == {
        "loss": 0.25,
        "accuracy": 0.9,
    }


def test_hydra_optuna_callback_on_job_end_syncs_trial_attributes(
    monkeypatch,
    tmp_path,
):
    hydra_cfg = SimpleNamespace(mode="RunMode.MULTIRUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    captured = {}

    def fake_sync(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        optimize_module,
        "_sync_multirun_trial_attributes",
        fake_sync,
    )

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
        score_file=str(tmp_path / "scores.json"),
    )
    # Simulate on_compose_config having resolved the score_file
    callback._resolved_score_file = str(tmp_path / "scores.json")
    cfg = OmegaConf.create(
        {
            "name": "demo",
            "experiment_name": "security_classification_linear_hsj",
            "optimizers": ["loss"],
            "directions": ["minimize"],
        },
    )

    callback.on_job_end(
        cfg,
        job_return=SimpleNamespace(
            return_value={"loss": 0.25, "accuracy": 0.9},
        ),
    )

    assert captured["hydra_cfg"] is hydra_cfg
    assert captured["score_payload"] == {
        "loss": 0.25,
        "accuracy": 0.9,
        "experiment_name": optimize_module.hash_conf_values(
            "security_classification_linear_hsj",
        ),
    }
    assert captured["optimizers"] == ["loss"]
    assert captured["directions"] == ["minimize"]
    assert captured["experiment_name"] == "security_classification_linear_hsj"


def test_hydra_optuna_callback_on_job_end_returns_without_files(monkeypatch):
    hydra_cfg = SimpleNamespace(mode="RunMode.MULTIRUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
    )

    # No files section: should no-op without raising.
    callback.on_job_end(OmegaConf.create({"name": "demo"}), job_return=None)


def test_hydra_optuna_callback_on_job_end_returns_without_score_file(monkeypatch):
    hydra_cfg = SimpleNamespace(mode="RunMode.MULTIRUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
    )

    # Files exists but score_file missing: should no-op without raising.
    callback.on_job_end(
        OmegaConf.create({"name": "demo", "files": {}}),
        job_return=SimpleNamespace(return_value={"loss": 0.5}),
    )


def test_hydra_optuna_callback_on_compose_config_uses_constructor_params_file(
    monkeypatch,
    tmp_path,
):
    hydra_cfg = SimpleNamespace(
        mode="RunMode.RUN",
        runtime=SimpleNamespace(output_dir=str(tmp_path)),
        job=SimpleNamespace(name="__main__"),
    )
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)
    monkeypatch.setattr(
        optimize_module,
        "_normalize_mode_cfg",
        lambda cfg, h, **kw: cfg,
    )
    monkeypatch.setattr(
        optimize_module,
        "_seed_experiment_uuid_for_current_trial",
        lambda **kw: None,
    )

    params_file = str(tmp_path / "params.yaml")
    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
        params_file=params_file,
    )
    cfg = OmegaConf.create({"name": "demo"})

    callback.on_compose_config(cfg)

    params_path = tmp_path / "params.yaml"
    assert params_path.exists()
    assert "name: demo" in params_path.read_text()


def test_hydra_optuna_callback_on_compose_config_resolves_single_run_paths_from_hydra_run_dir(
    monkeypatch,
    tmp_path,
):
    hydra_cfg = SimpleNamespace(
        mode="RunMode.RUN",
        run=SimpleNamespace(dir=str(tmp_path / "run_dir")),
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

    run_dir = tmp_path / "run_dir"
    assert cfg.files.log_file == str(run_dir / "optimize.log")
    assert cfg.files.score_file == str(run_dir / "scores.json")
    assert cfg.files.params_file == str(run_dir / "params.yaml")
    assert cfg.files.error_file == str(run_dir / "error.log")
    assert (run_dir / "params.yaml").exists()


def test_hydra_optuna_callback_on_job_end_uses_constructor_score_file(
    monkeypatch,
    tmp_path,
):
    hydra_cfg = SimpleNamespace(
        mode="RunMode.RUN",
        runtime=SimpleNamespace(output_dir=str(tmp_path)),
        job=SimpleNamespace(name="__main__"),
    )
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
        score_file=str(tmp_path / "scores.json"),
    )
    # Simulate on_compose_config having resolved the score_file
    callback._resolved_score_file = str(tmp_path / "scores.json")
    cfg = OmegaConf.create({"name": "demo"})

    callback.on_job_end(
        cfg,
        job_return=SimpleNamespace(return_value={"loss": 0.25, "accuracy": 0.9}),
    )

    score_path = tmp_path / "scores.json"
    assert score_path.exists()
    assert json.loads(score_path.read_text()) == {"loss": 0.25, "accuracy": 0.9}


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


def test_set_trial_attributes_single_trial_syncs_without_matching_hash(caplog):
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

    assert storage.attrs[(11, "accuracy")] == 0.9
    assert "Skipping trial attribute sync" not in caplog.text


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
            DummyTrial(
                number=2,
                trial_id=12,
                user_attrs={"experiment_name": "another_hash"},
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


def test_seed_experiment_uuid_for_current_trial_tags_trial(tmp_path):
    study_name = "seed_uuid"
    storage = f"sqlite:///{tmp_path / 'seed_uuid.db'}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.ask()

    hydra_cfg = OmegaConf.create(
        {
            "mode": "RunMode.MULTIRUN",
            "sweeper": {"storage": storage, "study_name": study_name},
            "job": {"id": "0"},
        },
    )

    optimize_module._seed_experiment_uuid_for_current_trial(
        hydra_cfg=hydra_cfg,
        experiment_name="security_classification_linear_hsj",
    )

    reloaded = optuna.load_study(study_name=study_name, storage=storage)
    trial = reloaded.get_trials(deepcopy=False)[0]
    assert "experiment_name" in trial.user_attrs


def test_get_hydra_job_identifier_prefers_job_id():
    hydra_cfg = SimpleNamespace(job=SimpleNamespace(id="17", num=5))
    assert optimize_module._get_hydra_job_identifier(hydra_cfg) == "17"


def test_get_hydra_job_identifier_normalizes_joblib_style_ids():
    hydra_cfg = SimpleNamespace(job=SimpleNamespace(id="__main___0", num=5))
    assert optimize_module._get_hydra_job_identifier(hydra_cfg) == "0"


def test_parallel_jobs_do_not_break_synchronization(tmp_path):
    study_name = "parallel_sync"
    storage = f"sqlite:///{tmp_path / 'parallel_sync.db'}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.ask()
    study.ask()

    hydra_cfg0 = OmegaConf.create(
        {
            "mode": "RunMode.MULTIRUN",
            "sweeper": {"storage": storage, "study_name": study_name},
            "job": {"id": "0"},
        },
    )
    hydra_cfg1 = OmegaConf.create(
        {
            "mode": "RunMode.MULTIRUN",
            "sweeper": {"storage": storage, "study_name": study_name},
            "job": {"id": "1"},
        },
    )

    optimize_module._seed_experiment_uuid_for_current_trial(
        hydra_cfg=hydra_cfg0,
        experiment_name="parallel_job_0",
    )
    optimize_module._seed_experiment_uuid_for_current_trial(
        hydra_cfg=hydra_cfg1,
        experiment_name="parallel_job_1",
    )

    optimize_module._sync_multirun_trial_attributes(
        hydra_cfg=hydra_cfg0,
        score_payload={"loss": 0.20, "accuracy": 0.81, "latency": 2.5},
        optimizers=["loss"],
        directions=["minimize"],
        experiment_name="parallel_job_0",
    )
    optimize_module._sync_multirun_trial_attributes(
        hydra_cfg=hydra_cfg1,
        score_payload={"loss": 0.10, "accuracy": 0.93, "latency": 1.1},
        optimizers=["loss"],
        directions=["minimize"],
        experiment_name="parallel_job_1",
    )

    reloaded = optuna.load_study(study_name=study_name, storage=storage)
    trials = sorted(reloaded.get_trials(deepcopy=False), key=lambda t: t.number)
    assert trials[0].user_attrs["accuracy"] == 0.81
    assert trials[1].user_attrs["accuracy"] == 0.93
    assert "experiment_name" in trials[0].user_attrs
    assert "experiment_name" in trials[1].user_attrs


def test_job_restarts_do_not_break_syncrhonization(tmp_path):
    study_name = "restart_sync"
    storage = f"sqlite:///{tmp_path / 'restart_sync.db'}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    study.ask()

    hydra_cfg = OmegaConf.create(
        {
            "mode": "RunMode.MULTIRUN",
            "sweeper": {"storage": storage, "study_name": study_name},
            "job": {"id": "0"},
        },
    )

    optimize_module._seed_experiment_uuid_for_current_trial(
        hydra_cfg=hydra_cfg,
        experiment_name="restart_case",
    )

    optimize_module._sync_multirun_trial_attributes(
        hydra_cfg=hydra_cfg,
        score_payload={"loss": 0.30, "accuracy": 0.70, "throughput": 10},
        optimizers=["loss"],
        directions=["minimize"],
        experiment_name="restart_case",
    )
    # Simulate restart/retry of same Hydra job id against same trial.
    optimize_module._sync_multirun_trial_attributes(
        hydra_cfg=hydra_cfg,
        score_payload={"loss": 0.20, "accuracy": 0.88, "throughput": 16},
        optimizers=["loss"],
        directions=["minimize"],
        experiment_name="restart_case",
    )

    reloaded = optuna.load_study(study_name=study_name, storage=storage)
    synced_trial = reloaded.get_trials(deepcopy=False)[0]
    assert synced_trial.user_attrs["accuracy"] == 0.88
    assert synced_trial.user_attrs["throughput"] == 16
    assert "experiment_name" in synced_trial.user_attrs


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


def test_optimize_main_executes_once_in_multirun(monkeypatch):
    class DummyBase:
        pass

    class DummyExperiment(DummyBase):
        def execute_without_mercy(self):
            captured["calls"] += 1
            return {"best": 0.1, "aux": 3.0}

    conf_obj = DummyExperiment()
    captured = {"calls": 0}

    def fake_instantiate(cfg):
        captured["cfg"] = cfg
        return conf_obj

    hydra_cfg = SimpleNamespace(
        mode="RunMode.MULTIRUN",
        sweeper={"storage": "sqlite:///db.sqlite3", "study_name": "demo-study"},
    )

    monkeypatch.setattr(optimize_module, "ConfigBase", DummyBase)
    monkeypatch.setattr(optimize_module, "ExperimentConfig", DummyExperiment)
    monkeypatch.setattr(optimize_module, "instantiate", fake_instantiate)
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    result = optimize_module.optimize_main(OmegaConf.create({"name": "demo"}))

    assert result == {"best": 0.1, "aux": 3.0}
    assert captured["calls"] == 1
    assert isinstance(captured["cfg"], dict)
    assert captured["cfg"]["_target_"] == "deckard.ExperimentConfig"
    assert "name" not in captured["cfg"]


def test_optimize_main_runs_hydra_configured_pytorch_experiment(monkeypatch):
    hydra_cfg = SimpleNamespace(mode="RunMode.RUN")
    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: hydra_cfg)

    X = torch.randn(48, 4)
    y = torch.randint(0, 2, (48,))
    dataset = torch.utils.data.TensorDataset(X, y)

    monkeypatch.setattr(
        "deckard.frameworks.pytorch.data.load_class",
        lambda *_args, **_kwargs: dataset,
    )

    cfg = OmegaConf.create(
        {
            "_target_": "deckard.experiment.ExperimentConfig",
            "library": "pytorch",
            "classifier": True,
            "data": {
                "_target_": "deckard.frameworks.pytorch.data.PytorchDataConfig",
                "dataset_name": "torch.utils.data.TensorDataset",
                "data_params": {},
                "train_size": 32,
                "test_size": 16,
                "stratify": True,
            },
            "model": {
                "_target_": "deckard.frameworks.pytorch.model.PytorchModelConfig",
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
    assert "accuracy" in scores["test"]
    assert "optimizer_loss" in scores


def test_callback_run_hooks_and_multirun_end_paths(monkeypatch):
    calls = []
    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
    )

    monkeypatch.setattr(
        optimize_module,
        "set_study_metric_names",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        optimize_module.HydraConfig,
        "get",
        lambda: SimpleNamespace(mode="RunMode.RUN"),
    )

    callback.on_run_start(OmegaConf.create({"name": "demo"}))
    callback.on_run_end(OmegaConf.create({"name": "demo"}))

    calls.clear()
    callback.study = None
    callback.on_multirun_end(OmegaConf.create({"name": "demo"}))
    assert calls == []

    callback.study = DummyStudy()
    callback.on_multirun_end(OmegaConf.create({"name": "demo"}))
    assert len(calls) == 1
    assert calls[0]["optimizers"] == ["loss"]


def test_mode_helpers_sweeper_assertions_and_extract_scores_paths():
    hydra_multirun = OmegaConf.create(
        {
            "mode": "RunMode.MULTIRUN",
            "sweeper": {
                "storage": "sqlite:///db.sqlite3",
                "study_name": "demo-study",
            },
            "sweep": {"dir": "outputs", "subdir": "demo/0"},
            "job": {"name": "optimize", "id": "0"},
        },
    )
    hydra_run = OmegaConf.create({"mode": "RunMode.RUN"})

    assert optimize_module._is_multirun_mode(hydra_multirun)
    assert not optimize_module._is_run_mode(hydra_multirun)
    assert optimize_module._is_run_mode(hydra_run)

    optimize_module._assert_multirun_sweeper(hydra_multirun)
    with pytest.raises(AssertionError, match="Sweeper must be specified"):
        optimize_module._assert_multirun_sweeper(
            OmegaConf.create({"mode": "RunMode.MULTIRUN"}),
        )

    resolved_paths = optimize_module._resolve_multirun_paths(hydra_multirun)
    assert resolved_paths["score_file"].endswith("scores.json")

    run_cfg = {"name": "demo"}
    assert optimize_module._normalize_mode_cfg(run_cfg, hydra_run) is run_cfg

    payload = optimize_module._extract_scores_from_job_end_kwargs(
        job_return={"return_value": OmegaConf.create({"loss": 0.1})},
    )
    assert payload == {"loss": 0.1}
    assert optimize_module._extract_scores_from_job_end_kwargs(job_return={}) is None


def test_sync_multirun_trial_attributes_non_dict_and_keyerror(monkeypatch):
    hydra_cfg = OmegaConf.create(
        {
            "mode": "RunMode.MULTIRUN",
            "sweeper": {
                "storage": "sqlite:///db.sqlite3",
                "study_name": "demo-study",
            },
            "job": {"id": "0"},
        },
    )

    optimize_module._sync_multirun_trial_attributes(
        hydra_cfg=hydra_cfg,
        score_payload="not-a-dict",
        optimizers=["loss"],
        directions=["minimize"],
        experiment_name="demo",
    )

    monkeypatch.setattr(
        optimize_module.optuna.study,
        "load_study",
        lambda **kwargs: (_ for _ in ()).throw(KeyError("missing")),
    )
    optimize_module._sync_multirun_trial_attributes(
        hydra_cfg=hydra_cfg,
        score_payload={"loss": 0.1, "latency": 3.0},
        optimizers=["loss"],
        directions=["minimize"],
        experiment_name="demo",
    )


def test_set_trial_attributes_trial_number_and_set_user_attr_fallback():
    class TrialWithSetter:
        def __init__(self):
            self.number = 2
            self.user_attrs = {}
            self.attrs = {}

        def set_user_attr(self, key, value):
            self.attrs[key] = value

    trial = TrialWithSetter()
    study = SimpleNamespace(
        study_name="demo-study",
        get_trials=lambda deepcopy=False: [trial],
    )

    optimize_module.set_trial_attributes(
        study,
        attrs={"accuracy": 0.99},
        experiment_name="explicit-demo",
        trial_number=2,
    )
    assert trial.attrs["accuracy"] == 0.99
    assert "experiment_name" in trial.attrs


def test_optimize_main_rejects_non_mapping_cfg(monkeypatch):
    monkeypatch.setattr(
        optimize_module.HydraConfig,
        "get",
        lambda: SimpleNamespace(mode="RunMode.RUN"),
    )

    with pytest.raises(AssertionError, match="cfg must resolve to a dictionary"):
        optimize_module.optimize_main(["not", "a", "dict"])


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
def test_deckard_optimize_hydra_multirun_cli_smoke(tmp_path):
    study_name = f"optimize_hydra_multirun_{uuid4().hex[:8]}"
    storage = f"sqlite:///{(tmp_path / 'optimize_hydra_multirun.db').as_posix()}"
    cmd = [
        sys.executable,
        "-m",
        "deckard",
        "optimize",
        "--multirun",
        "data=test-classification",
        "model=test-logistic",
        "attack=boundary",
        "defense=class-labels",
        "score=classification",
        "hydra.sweeper.n_trials=1",
        "hydra.sweeper.n_jobs=1",
        f"hydra.sweeper.study_name={study_name}",
        f"hydra.sweeper.storage={storage}",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(EXAMPLES_SKLEARN_DIR),
        env={
            **make_runtime_env(DECKARD_RC_PATH),
            "DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION": "1",
        },
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    run_dir = EXAMPLES_SKLEARN_DIR / "outputs" / "logs" / study_name / "0"
    assert run_dir.exists()
    assert (run_dir / ".hydra" / "config.yaml").exists()

    study = optuna.load_study(study_name=study_name, storage=storage)
    assert len(study.get_trials(deepcopy=False)) >= 1


def test_callback_job_start_and_end_guard_paths(monkeypatch):
    monkeypatch.setattr(
        optimize_module.HydraConfig,
        "get",
        lambda: SimpleNamespace(mode="RunMode.MULTIRUN"),
    )
    callback = optimize_module.OptunaStudyCallback(
        study_name="demo-study",
        storage="sqlite:///study.sqlite3",
        directions=["minimize"],
        optimizers=["loss"],
    )

    callback.on_job_start(OmegaConf.create({"name": "demo"}))
    callback.on_job_start(OmegaConf.create({"name": "demo", "files": {}}))

    callback.on_job_end(
        OmegaConf.create({"name": "demo", "files": {"score_file": "scores.json"}}),
        job_return=SimpleNamespace(return_value=None),
    )


def test_seed_experiment_uuid_no_matching_trial_and_no_storage(monkeypatch):
    study_no_match = SimpleNamespace(
        get_trials=lambda deepcopy=False: [SimpleNamespace(number=1, _trial_id=101)],
        _storage=DummyStorage(),
    )
    monkeypatch.setattr(
        optimize_module.optuna.study,
        "load_study",
        lambda **kwargs: study_no_match,
    )
    optimize_module._seed_experiment_uuid_for_current_trial(
        hydra_cfg=OmegaConf.create(
            {
                "sweeper": {
                    "storage": "sqlite:///db.sqlite3",
                    "study_name": "demo-study",
                },
                "job": {"id": "2"},
            },
        ),
        experiment_name="demo",
    )

    trial_without_storage = SimpleNamespace(number=0, _trial_id=None)
    study_without_storage = SimpleNamespace(
        get_trials=lambda deepcopy=False: [trial_without_storage],
    )
    monkeypatch.setattr(
        optimize_module.optuna.study,
        "load_study",
        lambda **kwargs: study_without_storage,
    )
    optimize_module._seed_experiment_uuid_for_current_trial(
        hydra_cfg=OmegaConf.create(
            {
                "sweeper": {
                    "storage": "sqlite:///db.sqlite3",
                    "study_name": "demo-study",
                },
                "job": {"id": "0"},
            },
        ),
        experiment_name="demo",
    )


def test_inject_experiment_name_non_dict_passthrough():
    assert optimize_module._inject_experiment_name("raw", "demo") == "raw"


def test_overwrite_frozen_trial_user_attr_paths(monkeypatch):
    assert not optimize_module._overwrite_frozen_trial_user_attr(
        SimpleNamespace(),
        trial_id=1,
        key="k",
        value="v",
    )

    class TrialModel:
        @staticmethod
        def find_or_raise_by_id(trial_id, session):
            return SimpleNamespace(id=trial_id)

    class TrialUserAttributeModel:
        def __init__(self, trial_id, key, value_json):
            self.trial_id = trial_id
            self.key = key
            self.value_json = value_json

        @staticmethod
        def find_by_trial_and_key(trial_model, key, session):
            _ = trial_model
            _ = key
            _ = session
            return None

    @contextlib.contextmanager
    def fake_scoped_session(_scoped_session, _commit):
        class Session:
            def __init__(self):
                self.added = []

            def add(self, item):
                self.added.append(item)

        yield Session()

    fake_models = SimpleNamespace(
        TrialModel=TrialModel,
        TrialUserAttributeModel=TrialUserAttributeModel,
    )
    monkeypatch.setattr(optimize_module, "_optuna_rdb_models", fake_models)
    monkeypatch.setattr(optimize_module, "_optuna_scoped_session", fake_scoped_session)

    study = SimpleNamespace(
        _storage=SimpleNamespace(_backend=SimpleNamespace(scoped_session=object())),
    )
    assert optimize_module._overwrite_frozen_trial_user_attr(study, 1, "k", "v")

    class FailingTrialModel:
        @staticmethod
        def find_or_raise_by_id(trial_id, session):
            _ = trial_id
            _ = session
            raise RuntimeError("boom")

    monkeypatch.setattr(
        optimize_module,
        "_optuna_rdb_models",
        SimpleNamespace(
            TrialModel=FailingTrialModel,
            TrialUserAttributeModel=TrialUserAttributeModel,
        ),
    )
    assert not optimize_module._overwrite_frozen_trial_user_attr(study, 1, "k", "v")


def test_mode_and_multirun_cfg_branches(tmp_path):
    hydra_cfg = OmegaConf.create(
        {
            "mode": "RunMode.MULTIRUN",
            "sweeper": {
                "storage": "sqlite:///db.sqlite3",
                "study_name": "demo-study",
            },
            "sweep": {"dir": str(tmp_path), "subdir": "trial_0"},
            "job": {"name": "optimize"},
        },
    )

    explicit_cfg = {"experiment_name": "already-here", "files": {}}
    prepared = optimize_module._prepare_multirun_cfg(
        explicit_cfg,
        hydra_cfg,
        include_file_paths=True,
    )
    assert len(prepared["experiment_name"]) == 32
    assert prepared["files"]["score_file"].endswith("scores.json")

    unknown_mode_cfg = {"name": "demo"}
    unknown_mode = SimpleNamespace(mode="RunMode.UNKNOWN")
    assert (
        optimize_module._normalize_mode_cfg(unknown_mode_cfg, unknown_mode)
        is unknown_mode_cfg
    )


def test_optimizer_policy_auto_configures_from_root_cfg(monkeypatch):
    captured = {}

    callback = optimize_module.DefaultOptimizerCallback()
    cfg = OmegaConf.create(
        {
            "directions": ["maximize", "diff"],
            "optimizers": ["accuracy", "delta"],
            "report_trial_attrs": False,
            "pruning_enabled": True,
            "dvclive_enabled": True,
        },
    )

    monkeypatch.setattr(optimize_module.HydraConfig, "get", lambda: SimpleNamespace(mode="RunMode.MULTIRUN"))
    monkeypatch.setattr(optimize_module, "_normalize_mode_cfg", lambda config, hydra_cfg, **kwargs: config)
    monkeypatch.setattr(optimize_module, "_seed_experiment_uuid_for_current_trial", lambda **kwargs: None)
    monkeypatch.setattr(optimize_module, "_resolve_multirun_paths", lambda hydra_cfg: {})
    monkeypatch.setattr(optimize_module, "_sync_multirun_trial_attributes", lambda **kwargs: captured.update(kwargs))

    callback.on_compose_config(cfg)
    callback._resolved_score_file = None
    callback.on_job_end(
        cfg,
        job_return=SimpleNamespace(return_value={"accuracy": 0.91, "delta": 0.04}),
    )

    assert callback.optimizers == ["accuracy", "delta"]
    assert callback.directions == ["maximize", "diff"]
    assert callback.optimizer.report_trial_attrs is False
    assert captured == {}


def test_stage_dependent_hash_payload_includes_selected_components():
    cfg = {
        "stage": "train",
        "data": {"name": "adult"},
        "model": {"name": "rf"},
        "attack": {"name": "hsj"},
        "score": {"name": "classification"},
        "directions": ["maximize"],
        "optimizers": ["accuracy"],
    }

    payload = optimize_module._build_stage_dependent_hash_payload(cfg)

    assert payload["stage"] == "train"
    assert "data" in payload["components"]
    assert "model" in payload["components"]
    assert "attack" not in payload["components"]


def test_stage_dependent_hash_payload_includes_attack_for_attack_stage():
    cfg = {
        "stage": "attack",
        "data": {"name": "adult"},
        "model": {"name": "rf"},
        "attack": {"name": "hsj"},
        "detector": {"name": "spectral"},
        "defense": {"name": "gaussian"},
    }

    payload = optimize_module._build_stage_dependent_hash_payload(cfg)

    assert payload["stage"] == "attack"
    assert "attack" in payload["components"]


def test_canonical_stage_components_include_attack_participation():
    assert "attack" in CANONICAL_EXPERIMENT_STAGE_COMPONENTS
    assert "attack" in CANONICAL_EXPERIMENT_STAGE_COMPONENTS["attack"]
    assert "attack" in CANONICAL_EXPERIMENT_STAGE_COMPONENTS["score"]


def test_canonical_stage_components_include_plugin_framework_plot_participants():
    assert "plugins" in CANONICAL_EXPERIMENT_STAGE_COMPONENTS["score"]
    assert "framework" in CANONICAL_EXPERIMENT_STAGE_COMPONENTS["score"]
    assert "plot" in CANONICAL_EXPERIMENT_STAGE_COMPONENTS["score"]


def test_extract_scores_kwargs_path_and_none():
    payload = optimize_module._extract_scores_from_job_end_kwargs(
        kwargs={"job_return": {"return_value": {"loss": 0.1}}},
    )
    assert payload == {"loss": 0.1}
    assert (
        optimize_module._extract_scores_from_job_end_kwargs(job_return=None, kwargs={})
        is None
    )


def test_sync_multirun_trial_attributes_missing_sweeper_values_and_empty_attrs():
    optimize_module._sync_multirun_trial_attributes(
        hydra_cfg=OmegaConf.create({"sweeper": {"study_name": "demo"}}),
        score_payload={"loss": 1.0},
        optimizers=["loss"],
        directions=["minimize"],
        experiment_name="demo",
    )
    optimize_module._sync_multirun_trial_attributes(
        hydra_cfg=OmegaConf.create(
            {"sweeper": {"storage": "sqlite:///db.sqlite3", "study_name": "demo"}},
        ),
        score_payload={"loss": 1.0},
        optimizers=["loss"],
        directions=["minimize"],
        experiment_name="demo",
    )


def test_execute_runtime_object_keeps_file_resolution_callback_owned(monkeypatch):
    conf = DummyConf()
    conf.files.log_file = ""
    conf.files.score_file = ""
    conf.files.params_file = ""
    conf.files.error_file = ""
    conf.optimizers = ["loss"]
    conf.directions = ["minimize"]
    conf.execute_without_mercy = lambda: {"loss": 0.2, "accuracy": 0.9}

    result = optimize_module.OptunaStudyCallback.execute_runtime_object(conf)

    assert result == {"loss": 0.2, "accuracy": 0.9}


def test_set_study_attributes_type_error():
    with pytest.raises(TypeError, match="attrs must be dict-like"):
        optimize_module.set_study_attributes(DummyStudy(), ["not", "dict"])


def test_optimize_main_accepts_plain_dict_cfg(monkeypatch):
    class DummyBase:
        def __call__(self):
            return {"ok": True}

    monkeypatch.setattr(optimize_module, "ConfigBase", DummyBase)
    monkeypatch.setattr(optimize_module, "instantiate", lambda cfg: DummyBase())
    monkeypatch.setattr(
        optimize_module.HydraConfig,
        "get",
        lambda: SimpleNamespace(mode="RunMode.RUN"),
    )

    result = optimize_module.optimize_main({"name": "demo"})
    assert result == {"ok": True}


def test_prepare_multirun_file_paths_without_to_dict(tmp_path):
    class BareConf:
        def __init__(self):
            self.files = DummyFiles(tmp_path)
            self.experiment_name = None
            self.post_init_calls = 0

        def __post_init__(self):
            self.post_init_calls += 1

        def __str__(self):
            return "bare-conf"

    conf = BareConf()
    hydra_cfg = SimpleNamespace(
        job=SimpleNamespace(name="optimize"),
        sweep=SimpleNamespace(dir=str(tmp_path), subdir="run_9"),
    )

    optimize_module.prepare_multirun_file_paths(hydra_cfg, conf)
    assert conf.post_init_calls == 1
    assert len(conf.experiment_name) == 32


def test_direction_and_objective_normalization_paths():
    assert optimize_module._normalize_direction("Direction.minimize") == "minimize"
    assert optimize_module._filter_optuna_objectives(None, "loss") == ([], ["loss"])
    assert optimize_module._filter_optuna_objectives(
        OmegaConf.create(["minimize"]),
        ("loss",),
    ) == (["minimize"], ["loss"])
    with pytest.raises(ValueError, match="Invalid direction"):
        optimize_module._normalize_direction("sideways")


def test_set_study_metric_names_rejects_invalid_optimizer_type():
    with pytest.raises(ValueError, match="optimizers must be a ListConfig"):
        optimize_module.set_study_metric_names(DummyStudy(), 123)


def test_set_trial_attributes_remaining_error_and_guard_paths(monkeypatch, caplog):
    study_no_trials = SimpleNamespace(
        study_name="demo-study",
        get_trials=lambda deepcopy=False: [],
    )
    optimize_module.set_trial_attributes(study_no_trials, {"a": 1}, "demo")
    assert "no trials found" in caplog.text

    with pytest.raises(TypeError, match="attrs must be a dict-like object"):
        optimize_module.set_trial_attributes(
            SimpleNamespace(
                study_name="demo-study",
                get_trials=lambda deepcopy=False: [SimpleNamespace(number=0)],
            ),
            attrs="bad",
            experiment_name="demo",
        )

    study_no_target = SimpleNamespace(
        study_name="demo-study",
        get_trials=lambda deepcopy=False: [
            DummyTrial(number=1, trial_id=11, user_attrs={"experiment_name": "x"}),
            DummyTrial(number=2, trial_id=12, user_attrs={"experiment_name": "y"}),
        ],
    )
    optimize_module.set_trial_attributes(study_no_target, {"a": 1}, "demo")

    class TrialNoSet:
        def __init__(self):
            self.number = 5
            self.trial_id = None
            self.user_attrs = {}

    study_no_handles = SimpleNamespace(
        study_name="demo-study",
        get_trials=lambda deepcopy=False: [TrialNoSet()],
    )
    with pytest.raises(RuntimeError, match="Unable to set trial attribute"):
        optimize_module.set_trial_attributes(
            study_no_handles,
            attrs={"meta": {"x": 1}},
            experiment_name="demo",
            trial_number=5,
        )

    class StorageRaises:
        def set_trial_user_attr(self, trial_id, key, value):
            _ = trial_id
            _ = key
            _ = value
            raise optuna.exceptions.UpdateFinishedTrialError("done")

    calls = {"overwrite": 0}
    monkeypatch.setattr(
        optimize_module,
        "_overwrite_frozen_trial_user_attr",
        lambda study, trial_id, key, value: calls.__setitem__(
            "overwrite",
            calls["overwrite"] + 1,
        )
        or True,
    )

    study_retry = SimpleNamespace(
        study_name="demo-study",
        _storage=StorageRaises(),
        get_trials=lambda deepcopy=False: [DummyTrial(number=0, trial_id=99)],
    )
    optimize_module.set_trial_attributes(
        study_retry,
        attrs={"meta": OmegaConf.create({"k": 1})},
        experiment_name="demo",
        trial_number=0,
    )
    assert calls["overwrite"] >= 1


def test_save_params_file_accepts_dictconfig(tmp_path):
    cfg = OmegaConf.create({"params": {"lr": 0.1}, "trainer": {"epochs": 2}})
    files = {"params_file": str(tmp_path / "params.yaml")}

    result = optimize_module.save_params_file(cfg, files)
    assert result.trainer.epochs == 2


def test_filter_scores_missing_min_max_and_no_directions_paths():
    values, attrs = optimize_module.filter_scores(
        scores={"other": 1.0},
        optimizers=["loss", "accuracy"],
        directions=["minimize", "maximize"],
    )
    assert isinstance(values, tuple)
    assert values[0] == float("inf")
    assert values[1] == float("-inf")
    assert attrs["other"] == 1.0

    values_no_dir, attrs_no_dir = optimize_module.filter_scores(
        scores={"loss": 0.2, "latency": 3.0},
        optimizers=["loss"],
        directions=[],
    )
    assert values_no_dir == 0.2
    assert attrs_no_dir["latency"] == 3.0
