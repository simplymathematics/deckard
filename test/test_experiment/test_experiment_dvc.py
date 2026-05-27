from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import types

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

import deckard.experiment.dvc as dvc_module
from deckard.experiment import ExperimentConfig
from deckard.experiment.dvc import (
    DVCExperimentConfig,
    build_dvc_experiment_plugin_hooks,
    build_dvc_cmd,
    build_dvc_stage_name,
    build_dvc_stage_plan,
    render_dvclive_report,
    generate_vega_lite_plot_spec,
    generate_dvc_pipeline,
    run_dvc_experiment_plugin_hook,
)
from deckard.experiment.repro import run_repro_experiment_plugin_hook
from deckard.file import FileConfig
from deckard.score.base import ScorerConfig
from deckard.score.dvc import DVCSystemScorerDictConfig, dvc_component_stats_score


HAS_DVCLIVE = importlib.util.find_spec("dvclive") is not None


def _make_runtime_env(rc_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    if rc_path.exists():
        for raw_line in rc_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or not line.startswith("export "):
                continue
            key_value = line[len("export ") :]
            if "=" not in key_value:
                continue
            key, value = key_value.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    env.setdefault("DECKARD_TEST_MAX_SAMPLES", "200")
    env.setdefault("MPLBACKEND", "Agg")
    return env


def _make_experiment_stub(*, name: str = "demo-exp", with_files: bool = True):
    files = None
    if with_files:
        files = FileConfig(
            params_file="outputs/demo/params.yaml",
            score_file="outputs/demo/scores.json",
            log_file="outputs/demo/run.log",
            error_file="outputs/demo/error.log",
        )
    return SimpleNamespace(
        experiment_name=name,
        library="sklearn",
        classifier=True,
        evaluation_mode="standard",
        score_mode="test",
        random_state=1,
        files=files,
        data=None,
        model=None,
        defense=None,
        attack=None,
        detector=None,
        score=None,
    )


def _make_real_experiment_from_examples(tmp_path: Path) -> ExperimentConfig:
    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
    ).as_posix()
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="default",
            overrides=[
                "+stage=persist",
                "files=default",
                f"+files.params_file={tmp_path.as_posix()}/params.yaml",
                f"+files.score_file={tmp_path.as_posix()}/scores.json",
                f"+files.log_file={tmp_path.as_posix()}/run.log",
                f"+files.error_file={tmp_path.as_posix()}/error.log",
                "hydra/job_logging=none",
                "hydra/hydra_logging=none",
            ],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    score_cfg = resolved.get("score")
    if (
        isinstance(score_cfg, dict)
        and score_cfg.get("_target_") == "deckard.score.base.DefaultClassifierConfig"
    ):
        score_cfg["_target_"] = "deckard.score.base.DefaultClassifierScorerDictConfig"
    allowed = set(inspect.signature(ExperimentConfig).parameters.keys())
    payload: dict[str, object] = {"_target_": "deckard.ExperimentConfig"}
    for key, value in resolved.items():
        if key in allowed:
            payload[key] = value
    exp = instantiate(payload)
    assert isinstance(exp, ExperimentConfig)
    return exp


def test_build_dvc_stage_name_uses_canonical_shape():
    assert build_dvc_stage_name("experiment", "persist") == "experiment__persist"
    assert build_dvc_stage_name("model", "train") == "model__train"


def test_stage_plan_defaults_include_canonical_stage_names():
    exp = _make_experiment_stub()
    plan = build_dvc_stage_plan(exp)
    names = [entry["name"] for entry in plan]
    assert names == [
        "data__load",
        "data__sample",
        "data__pipeline",
        "experiment__score",
        "experiment__persist",
    ]


def test_stage_plan_full_components_include_detector_train_and_attack_generation():
    exp = _make_experiment_stub()
    exp.model = object()
    exp.defense = object()
    exp.attack = object()
    exp.detector = object()

    plan = build_dvc_stage_plan(exp)
    names = [entry["name"] for entry in plan]
    required = [
        "data__load",
        "data__sample",
        "data__pipeline",
        "defense__apply-fit-defense",
        "model__train",
        "defense__apply-predict-defense",
        "attack__generation",
        "detector__train",
        "detector__defense",
        "experiment__score",
        "experiment__persist",
    ]
    for prefix in required:
        assert any(name.startswith(prefix) for name in names)

    def _first_index(prefix: str) -> int:
        return next(i for i, name in enumerate(names) if name.startswith(prefix))

    assert (
        _first_index("attack__generation")
        < _first_index("detector__train")
        < _first_index("detector__defense")
    )


def test_real_experiment_stage_order_places_attack_before_detector(tmp_path: Path):
    exp = _make_real_experiment_from_examples(tmp_path)
    exp.detector = object()

    plan = build_dvc_stage_plan(exp, mode="single")
    names = [entry["name"] for entry in plan]

    attack_idx = next(
        i for i, name in enumerate(names) if name.startswith("attack__generation")
    )
    detector_train_idx = next(
        i for i, name in enumerate(names) if name.startswith("detector__train")
    )
    detector_defense_idx = next(
        i for i, name in enumerate(names) if name.startswith("detector__defense")
    )
    assert attack_idx < detector_train_idx < detector_defense_idx


def test_hook_style_stage_selection_maps_to_component_specific_names():
    exp = _make_experiment_stub()
    plan = build_dvc_stage_plan(
        exp,
        stage_selection=["before_score", "after_attack", "after_defense"],
    )
    names = [entry["name"] for entry in plan]
    assert names == ["experiment__score"]


def test_data_score_stage_decomposes_by_stage_group_not_metric_key():
    exp = _make_experiment_stub(with_files=True)
    exp.data = SimpleNamespace(
        scorer=SimpleNamespace(
            stage=[],
            scorers={
                "num_classes": SimpleNamespace(stage=["post-sample"]),
                "class_count_min": SimpleNamespace(stage=["post-sample"]),
            },
        ),
    )

    plan = build_dvc_stage_plan(
        exp,
        stage_selection=["data-score"],
        include_cache_aliases=False,
    )
    names = [entry["name"] for entry in plan]
    assert names == ["data__data-score-post-sample"]


def test_model_score_stage_decomposes_by_stage_group_not_metric_key():
    exp = _make_experiment_stub(with_files=True)
    exp.model = SimpleNamespace(
        scorer=SimpleNamespace(
            stage=[],
            scorers={
                "accuracy": SimpleNamespace(stage=["test"]),
                "precision": SimpleNamespace(stage=["test"]),
            },
        ),
    )

    plan = build_dvc_stage_plan(
        exp,
        stage_selection=["model-score"],
        include_cache_aliases=False,
    )
    names = [entry["name"] for entry in plan]
    assert names == ["model__model-score-test"]


def test_attack_score_stage_decomposes_by_scorer_stage_group_not_attack_alias():
    exp = _make_experiment_stub(with_files=True)
    attack_cfg = SimpleNamespace(
        alias="hsj",
        scorer=SimpleNamespace(
            stage=[],
            _profile_attr="evasion",
            scorers={
                "accuracy": SimpleNamespace(stage=["adversarial"]),
                "f1": SimpleNamespace(stage=["adversarial"]),
            },
        ),
    )
    exp.attack = attack_cfg
    exp._attack_chain = [attack_cfg]

    plan = build_dvc_stage_plan(
        exp,
        stage_selection=["attack-score"],
        include_cache_aliases=False,
    )
    names = [entry["name"] for entry in plan]
    assert names == ["attack__attack-score-adversarial"]


def test_persist_stage_uses_experiment_identity_in_single_mode_without_files():
    exp = _make_experiment_stub(name="My Experiment", with_files=True)
    plan = build_dvc_stage_plan(exp, stage_selection=["persist"], mode="single")
    persist = plan[0]
    assert persist["identity"] == "my-experiment"
    assert "outputs/logs/my-experiment" in persist["outs"][0]
    assert persist["plots"] == []


def test_generate_vega_lite_plot_spec_writes_file(tmp_path: Path):
    output_file = tmp_path / "plots" / "roc_auc.yaml"
    payload = generate_vega_lite_plot_spec(
        output_file=output_file.as_posix(),
        title="ROC AUC by Threshold",
        x_field="threshold",
        y_field="roc_auc",
        color_field="split",
    )

    assert payload["output_file"].endswith("roc_auc.vl.json")
    assert Path(payload["output_file"]).exists()
    assert payload["title"] == "ROC AUC by Threshold"
    assert payload["encoding"]["x"]["field"] == "threshold"
    assert payload["encoding"]["y"]["field"] == "roc_auc"


def test_persist_stage_uses_hash_identity_in_multirun_mode_without_files():
    exp = _make_experiment_stub(name="ignored-name", with_files=True)
    plan = build_dvc_stage_plan(exp, stage_selection=["persist"], mode="multirun")
    persist = plan[0]
    assert persist["identity"] != "ignored-name"
    assert len(persist["identity"]) == 12
    assert persist["identity"].isalnum()


def test_persist_stage_metrics_policy_defaults_to_file_only_metrics():
    exp = _make_experiment_stub(with_files=True)
    plan = build_dvc_stage_plan(exp, stage_selection=["persist"], mode="single")
    persist = plan[0]

    assert persist["metrics"] == ["outputs/demo/scores.json"]


def test_vega_plot_metric_token_uses_first_configured_optimizer():
    exp = _make_experiment_stub(with_files=True)
    exp.optimizers = ["macro_f1"]
    attack_cfg = SimpleNamespace(alias="hsj", attack_params={"max_iter": 20})
    defense_cfg = SimpleNamespace(
        alias="class-labels",
        defense_params={"apply_fit": True},
    )

    params_manifest = {
        "runtime_kwargs": {
            "attack_cfg": attack_cfg,
            "defense_cfg": defense_cfg,
        },
    }
    tokens = dvc_module._resolve_plot_filename_tokens(exp, params_manifest)

    assert tokens["metric"] == "macro_f1"


def test_generate_dvc_pipeline_writes_yaml(tmp_path: Path):
    exp = _make_experiment_stub(with_files=True)
    output_file = tmp_path / "generated.dvc.yaml"
    params_file = tmp_path / "generated.params.yaml"

    payload = generate_dvc_pipeline(
        exp,
        output_file=output_file.as_posix(),
        params_file=params_file.as_posix(),
        stage_selection=["persist"],
        mode="multirun",
        overwrite=True,
    )

    assert output_file.exists()
    assert params_file.exists()
    assert "stages" in payload
    assert "params_payload" in payload
    assert "experiment__persist" in payload["stages"]


def test_generate_dvc_pipeline_emits_stage_param_key_paths(tmp_path: Path):
    exp = _make_experiment_stub(with_files=True)
    output_file = tmp_path / "generated.params.dvc.yaml"

    payload = generate_dvc_pipeline(
        exp,
        output_file=output_file.as_posix(),
        stage_selection=["load"],
        params_file="params.yaml",
        overwrite=True,
    )

    stage = payload["stages"]["data__load"]
    assert "params" in stage
    assert isinstance(stage["params"], list)
    assert isinstance(stage["params"][0], dict)
    assert "params.yaml" in stage["params"][0]
    key_paths = stage["params"][0]["params.yaml"]
    assert isinstance(key_paths, list)
    assert "data" in key_paths
    assert "model" not in key_paths


def test_persist_command_does_not_use_external_mkdir_prefix():
    exp = _make_experiment_stub(with_files=True)
    plan = build_dvc_stage_plan(exp, stage_selection=["persist"], mode="single")
    persist = plan[0]

    cmd = build_dvc_cmd(exp, persist, mode="single")
    assert "mkdir -p" not in cmd
    assert cmd.startswith("deckard optimize +stage=persist")


def test_multirun_mode_does_not_emit_hydra_multirun_flags():
    exp = _make_experiment_stub(with_files=True)
    plan = build_dvc_stage_plan(exp, stage_selection=["persist"], mode="multirun")
    persist = plan[0]

    cmd = build_dvc_cmd(exp, persist, mode="multirun", multirun_count=4)
    assert "--multirun" not in cmd
    assert "hydra.sweeper.n_trials" not in cmd
    assert cmd.startswith("deckard optimize +stage=persist")


@pytest.mark.parametrize("overwrite", [False, True])
def test_generate_dvc_pipeline_overwrite_behavior(tmp_path: Path, overwrite: bool):
    exp = _make_experiment_stub(with_files=True)
    output_file = tmp_path / "existing.dvc.yaml"
    output_file.write_text("stages: {}\n", encoding="utf-8")

    if overwrite:
        generate_dvc_pipeline(exp, output_file=output_file.as_posix(), overwrite=True)
        assert output_file.exists()
    else:
        with pytest.raises(FileExistsError):
            generate_dvc_pipeline(
                exp,
                output_file=output_file.as_posix(),
                overwrite=False,
            )


def test_stage_plan_raises_for_unknown_stage_token():
    exp = _make_experiment_stub(with_files=True)
    with pytest.raises(ValueError, match="Unsupported stage token"):
        build_dvc_stage_plan(exp, stage_selection=["bogus_stage"])


def test_stage_plan_raises_for_unsupported_mode():
    exp = _make_experiment_stub(with_files=True)
    with pytest.raises(ValueError, match="Unsupported run mode"):
        build_dvc_stage_plan(exp, mode="batch")


def test_score_stage_requires_score_file_alias():
    exp = _make_experiment_stub(with_files=True)
    exp.files.update(score_file=None)
    with pytest.raises(ValueError, match="Stage 'score' requires files.score_file"):
        build_dvc_stage_plan(
            exp,
            stage_selection=["score"],
            include_cache_aliases=False,
        )


def test_persist_stage_requires_explicit_file_aliases():
    exp = _make_experiment_stub(with_files=False)
    with pytest.raises(ValueError, match="Stage 'persist' requires file aliases"):
        build_dvc_stage_plan(
            exp,
            stage_selection=["persist"],
            include_cache_aliases=False,
        )


def test_cache_aliases_require_params_file_alias():
    exp = _make_experiment_stub(with_files=True)
    exp.files.update(params_file=None)
    with pytest.raises(
        ValueError,
        match="include_cache_aliases=True requires files.params_file",
    ):
        build_dvc_stage_plan(
            exp,
            stage_selection=["train"],
            include_cache_aliases=True,
        )


def test_build_dvc_experiment_plugin_hooks_adds_first_and_last_wrappers():
    first_hooks, last_hooks = build_dvc_experiment_plugin_hooks({"enabled": True})
    assert len(first_hooks) > 0
    assert len(last_hooks) > 0
    assert first_hooks[0].method_name == "_dvc_experiment_plugin_hook"
    assert last_hooks[-1].method_name == "_dvc_experiment_plugin_hook"
    assert first_hooks[0].method_kwargs["plugin_position"] == "first"
    assert last_hooks[0].method_kwargs["plugin_position"] == "last"


def test_run_dvc_and_repro_hooks_split_persistence_and_monitoring(monkeypatch):
    exp = _make_experiment_stub(with_files=True)
    exp.outputs = {}

    called = {"render": 0, "pull": 0, "push": 0}

    def _fake_render(_experiment, *, plugin):
        called["render"] += 1
        assert plugin.enabled is True
        return {
            "enabled": True,
            "dvclive_dir": "outputs/logs/demo/dvclive",
            "summary_json": "outputs/logs/demo/dvclive/summary.json",
            "report_html": "outputs/logs/demo/dvclive/report.html",
        }

    def _fake_pull(_experiment, _plugin):
        called["pull"] += 1
        return {"ok": True}

    def _fake_push(_experiment, _plugin):
        called["push"] += 1
        return {"ok": True}

    monkeypatch.setattr("deckard.experiment.dvc.render_dvclive_report", _fake_render)
    monkeypatch.setattr("deckard.experiment.dvc._run_dvc_pull", _fake_pull)
    monkeypatch.setattr("deckard.experiment.dvc._run_dvc_push", _fake_push)

    class _FakeLive:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def log_params(self, _params):
            return None

        def log_metric(self, _name, _value):
            return None

        def next_step(self):
            return None

        def end(self):
            return None

    monkeypatch.setitem(sys.modules, "dvclive", types.SimpleNamespace(Live=_FakeLive))

    before_load = run_repro_experiment_plugin_hook(
        exp,
        repro_plugin={"enabled": True},
        plugin_position="first",
        component="data",
        stage="load",
        event="before",
    )
    assert before_load["executed"] is True
    assert called["pull"] == 1

    skipped = run_dvc_experiment_plugin_hook(
        exp,
        dvc_plugin={"enabled": True},
        plugin_position="first",
        component="experiment",
        stage="persist",
        event="after",
    )
    assert skipped["executed"] is False
    assert called["render"] == 0

    executed = run_dvc_experiment_plugin_hook(
        exp,
        dvc_plugin={"enabled": True},
        plugin_position="last",
        component="experiment",
        stage="persist",
        event="after",
    )
    assert executed["executed"] is True
    assert called["render"] == 1

    persisted = run_repro_experiment_plugin_hook(
        exp,
        repro_plugin={"enabled": True},
        plugin_position="last",
        component="experiment",
        stage="persist",
        event="after",
    )
    assert persisted["executed"] is True
    assert called["push"] == 1


def test_run_dvc_plugin_hook_writes_structured_params_yaml(
    tmp_path: Path,
    monkeypatch,
):
    files = FileConfig(
        params_file=(tmp_path / "params.yaml").as_posix(),
        score_file=(tmp_path / "scores.json").as_posix(),
        log_file=(tmp_path / "run.log").as_posix(),
        error_file=(tmp_path / "error.log").as_posix(),
    )
    exp = SimpleNamespace(
        experiment_name="demo-exp",
        library="sklearn",
        classifier=True,
        evaluation_mode="standard",
        score_mode="test",
        random_state=1,
        files=files,
        outputs={},
        data=None,
        model=None,
        defense=None,
        attack=None,
        detector=None,
        score=None,
    )

    class _FakeLive:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def log_params(self, _params):
            return None

        def end(self):
            return None

    monkeypatch.setitem(sys.modules, "dvclive", types.SimpleNamespace(Live=_FakeLive))
    monkeypatch.setattr(
        "deckard.experiment.dvc._safe_run_dvc_cmd",
        lambda plugin, command, cwd=None: {"ok": True, "command": command},
    )

    result = run_repro_experiment_plugin_hook(
        exp,
        repro_plugin={
            "enabled": True,
            "mode": "single",
            "dvclive_dir": (tmp_path / "dvclive").as_posix(),
        },
        plugin_position="first",
        component="data",
        stage="load",
        event="before",
    )

    params_path = Path(exp.files.params_file)
    assert params_path.exists()
    payload = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    assert payload["_target_"] == "deckard.experiment.dvc.DVCExperimentConfig"
    assert payload["experiment"]["experiment_name"] == "demo-exp"
    assert payload["experiment"]["_target_"] == "deckard.experiment.ExperimentConfig"
    assert "dvc_plugin" not in payload["experiment"]
    assert isinstance(payload["dvc_plugin"], dict)
    dvclive_dir = payload["dvc_plugin"].get("dvclive_dir")
    if dvclive_dir not in [None, ""]:
        assert not Path(dvclive_dir).is_absolute()
    assert payload["_dvc"]["stage_selection"] == "load"
    assert payload["_dvc"]["run_mode"] == "single"
    assert isinstance(payload["_dvc"]["params_manifest"], dict)
    assert payload["experiment"]["dvclive_enabled"] is True
    assert payload["_dvc"]["dvclive"]["enabled"] is True
    assert payload["_dvc"]["pruning"]["enabled"] is False
    assert payload["_dvc"]["pruning"]["active"] is False
    assert result["params_file"] == params_path.as_posix()


def test_generate_dvc_pipeline_writes_deterministic_params_yaml(tmp_path: Path):
    exp = _make_experiment_stub(with_files=True)
    exp.pruning_enabled = True
    exp.hydra = {
        "sweeper": {
            "pruner": {
                "_target_": "optuna.pruners.MedianPruner",
            },
        },
    }
    exp.dvc_plugin = {"enabled": True, "mode": "multirun"}

    dvc_file = tmp_path / "pipeline.dvc.yaml"
    params_file = tmp_path / "pipeline.params.yaml"

    first = generate_dvc_pipeline(
        exp,
        output_file=dvc_file.as_posix(),
        params_file=params_file.as_posix(),
        stage_selection=["persist"],
        mode="multirun",
        overwrite=True,
    )
    first_text = params_file.read_text(encoding="utf-8")

    second = generate_dvc_pipeline(
        exp,
        output_file=dvc_file.as_posix(),
        params_file=params_file.as_posix(),
        stage_selection=["persist"],
        mode="multirun",
        overwrite=True,
    )
    second_text = params_file.read_text(encoding="utf-8")

    assert first_text == second_text
    payload = yaml.safe_load(first_text)
    assert payload["experiment"]["dvclive_enabled"] is True
    assert payload["_dvc"]["stage_selection"] == "persist"
    assert payload["_dvc"]["run_mode"] == "multirun"
    assert payload["_dvc"]["pruning"]["enabled"] is False
    assert payload["_dvc"]["pruning"]["pruner_configured"] is False
    assert payload["_dvc"]["pruning"]["active"] is False
    assert first["params_payload"] == second["params_payload"]


def test_log_dvclive_artifacts_prefers_log_image_for_images(
    tmp_path: Path,
    monkeypatch,
):
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"png")

    seen: dict[str, list[tuple[str, str]]] = {"images": [], "artifacts": []}

    class _FakeLive:
        def log_image(self, name: str, path: str):
            seen["images"].append((name, path))

        def log_artifact(self, path: str):
            seen["artifacts"].append((Path(path).name, path))

    exp = _make_experiment_stub(with_files=True)
    plugin = dvc_module.DVCExperimentPlugin(enabled=True)

    monkeypatch.setattr(
        dvc_module,
        "_ensure_live_instance",
        lambda experiment, plugin: _FakeLive(),
    )
    monkeypatch.setattr(
        dvc_module,
        "_resolve_stage_outputs",
        lambda experiment, stage, include_identity_dir=False, plugin=None: [
            image_path.as_posix(),
        ],
    )

    dvc_module._log_dvclive_artifacts_and_plots(exp, plugin)

    assert seen["images"] == [("chart.png", image_path.as_posix())]
    assert seen["artifacts"] == []


def test_dvc_experiment_config_wraps_experiment_and_plugin(tmp_path: Path):
    exp = _make_real_experiment_from_examples(tmp_path)
    wrapped = DVCExperimentConfig(
        experiment=exp,
        dvc_plugin={"enabled": True, "report_mode": "md"},
    )

    assert wrapped._target_ == "deckard.experiment.dvc.DVCExperimentConfig"
    assert isinstance(wrapped.to_experiment_config(), ExperimentConfig)
    assert wrapped.to_experiment_config().dvc_plugin["enabled"] is True
    assert wrapped.to_experiment_config().dvc_plugin["report_mode"] == "md"


def test_score_stage_power_hooks_merge_into_component_score_dicts(tmp_path: Path):
    exp = _make_real_experiment_from_examples(tmp_path)
    attack_cfg = SimpleNamespace(alias="fgsm", score_dict={})
    exp.data = SimpleNamespace(score_dict={})
    exp.model = SimpleNamespace(score_dict={})
    exp.attack = attack_cfg
    exp._attack_chain = [attack_cfg]
    exp.detector = SimpleNamespace(score_dict={})
    exp.score_dict = {}

    def _log_power_score(*, namespace: str, **kwargs):
        return {f"power/{namespace}/total_watts": 42.0}

    exp._log_power_score = _log_power_score
    exp._initialize_hook_orchestration()

    exp._run_experiment_stage_hooks("after", "data_score", component="data")
    exp._run_experiment_stage_hooks("after", "model_score", component="model")
    exp._run_experiment_stage_hooks("after", "attack_score", component="attack:fgsm")
    exp._run_experiment_stage_hooks("after", "detector_score", component="detector")

    assert exp.data.score_dict["power/data/total_watts"] == 42.0
    assert exp.model.score_dict["power/model/total_watts"] == 42.0
    assert attack_cfg.score_dict["power/attack/total_watts"] == 42.0
    assert exp.detector.score_dict["power/detector/total_watts"] == 42.0
    assert exp.score_dict["power/detector/total_watts"] == 42.0


def test_run_dvc_plugin_hook_adds_dvc_system_scores_after_component_score_stage(
    monkeypatch,
):
    exp = _make_experiment_stub(with_files=True)
    exp.outputs = {}
    exp.score_dict = {}
    exp.data = SimpleNamespace(score_dict={"data_metric": 1.0})

    monkeypatch.setattr(
        dvc_module,
        "_collect_system_monitor_scores",
        lambda experiment, plugin: {"system_monitor/cpu": 5.0},
    )

    result = run_dvc_experiment_plugin_hook(
        exp,
        dvc_plugin={"enabled": True},
        plugin_position="last",
        component="data",
        stage="data_score",
        event="after",
    )

    assert result["executed"] is True
    assert "dvc_system_scores" in result
    assert exp.score_dict["test"]["data_cpu"] == pytest.approx(5.0)


def test_run_dvc_plugin_hook_skips_custom_stage_outside_dvc_system_stages(
    monkeypatch,
):
    exp = _make_experiment_stub(with_files=True)
    exp.outputs = {}
    exp.score_dict = {}
    exp.data = SimpleNamespace(score_dict={"data_metric": 1.0})
    exp._dvc_system_scorer = DVCSystemScorerDictConfig(
        scorers={
            "data": ScorerConfig(
                score_name="data",
                score_function=dvc_component_stats_score,
                needs_labels=False,
                stage=["val"],
                score_params={"component": "data"},
            ),
        },
    )

    monkeypatch.setattr(
        dvc_module,
        "_collect_system_monitor_scores",
        lambda experiment, plugin: {"system_monitor/memory": 8.0},
    )

    result = run_dvc_experiment_plugin_hook(
        exp,
        dvc_plugin={"enabled": True},
        plugin_position="last",
        component="data",
        stage="val",
        event="after",
    )

    assert result["executed"] is False
    assert "dvc_system_error" not in result
    assert exp.score_dict == {}


@pytest.mark.skipif(
    not HAS_DVCLIVE,
    reason="requires dvclive installation",
)
def test_run_dvc_plugin_hook_skips_dvc_system_scorer_for_non_score_stage():
    exp = _make_experiment_stub(with_files=True)
    exp.outputs = {}
    exp.score_dict = {}

    result = run_dvc_experiment_plugin_hook(
        exp,
        dvc_plugin={"enabled": True},
        plugin_position="last",
        component="experiment",
        stage="persist",
        event="after",
    )

    assert result["executed"] is True
    assert "dvc_system_error" not in result


def test_collect_system_monitor_scores_falls_back_to_metrics_json(tmp_path: Path):
    exp = _make_experiment_stub(with_files=False)
    exp._dvc_plugin_runtime = {"dir": tmp_path.as_posix()}

    metrics_payload = {
        "step": 1,
        "system/cpu/usage (%)": 17.5,
        "system/ram/usage (%)": 33.0,
    }
    (tmp_path / "metrics.json").write_text(
        json.dumps(metrics_payload),
        encoding="utf-8",
    )

    plugin = dvc_module.coerce_dvc_experiment_plugin(
        {"enabled": True, "dvclive_dir": tmp_path.as_posix()},
    )
    collected = dvc_module._collect_system_monitor_scores(exp, plugin)

    assert collected["system_monitor/cpu/usage (%)"] == pytest.approx(17.5)
    assert collected["system_monitor/ram/usage (%)"] == pytest.approx(33.0)


def test_collect_system_monitor_scores_reads_live_runtime_monitor_state(
    tmp_path: Path,
):
    exp = _make_experiment_stub(with_files=False)
    exp._dvc_plugin_runtime = {
        "dir": tmp_path.as_posix(),
        "live": types.SimpleNamespace(
            _system_monitor=types.SimpleNamespace(
                _metrics={
                    "system/cpu/usage (%)": 21.0,
                    "system/ram/usage (%)": 44.0,
                },
            ),
        ),
    }

    plugin = dvc_module.coerce_dvc_experiment_plugin(
        {"enabled": True, "dvclive_dir": tmp_path.as_posix()},
    )
    collected = dvc_module._collect_system_monitor_scores(exp, plugin)

    assert collected["system_monitor/cpu/usage (%)"] == pytest.approx(21.0)
    assert collected["system_monitor/ram/usage (%)"] == pytest.approx(44.0)


def test_experiment_hash_payload_excludes_dvc_plugin(tmp_path: Path):
    exp = _make_real_experiment_from_examples(tmp_path)
    payload_without = exp.to_dict(for_hash=True)
    assert "dvc_plugin" not in payload_without

    exp.dvc_plugin = {
        "enabled": True,
        "mode": "single",
        "dvclive_dir": (tmp_path / "dvclive").as_posix(),
    }
    payload_with = exp.to_dict(for_hash=True)
    assert "dvc_plugin" not in payload_with


@pytest.mark.parametrize(
    "mode,expected_name",
    [
        ("md", "report.md"),
        ("notebook", "report.ipynb"),
        ("html", "report.html"),
    ],
)
def test_render_dvclive_report_mode_to_file_mapping(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    expected_name: str,
):
    class _FakeLive:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def make_summary(self):
            out_dir = Path(self.kwargs["dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "summary.json").write_text("{}", encoding="utf-8")

        def make_report(self):
            out_dir = Path(self.kwargs["dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            report_name = {
                "md": "report.md",
                "notebook": "report.ipynb",
                "html": "report.html",
                None: "report.html",
            }[self.kwargs.get("report")]
            (out_dir / report_name).write_text("report", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "dvclive", types.SimpleNamespace(Live=_FakeLive))

    exp = _make_experiment_stub(with_files=True)
    result = render_dvclive_report(
        exp,
        plugin={
            "enabled": True,
            "dvclive_dir": (tmp_path / "dvclive").as_posix(),
            "report_mode": mode,
            "make_summary": True,
            "make_report": True,
        },
    )

    assert result["report_mode"] == mode
    assert not Path(result["dvclive_dir"]).is_absolute()
    assert result["summary_json"] is not None
    assert not Path(result["summary_json"]).is_absolute()
    assert result["report_file"] is not None
    assert not Path(result["report_file"]).is_absolute()
    assert result["report_file"].endswith(expected_name)
    if mode == "html":
        assert result["report_html"] == result["report_file"]
        assert not Path(result["report_html"]).is_absolute()
    else:
        assert result["report_html"] is None


def test_disable_dvclive_gpu_monitor_when_nvml_unavailable(monkeypatch):
    fake_monitor = types.SimpleNamespace(
        GPU_AVAILABLE=True,
        nvmlInit=lambda: (_ for _ in ()).throw(RuntimeError("nvml missing")),
        nvmlShutdown=lambda: None,
    )

    def _fake_import(name: str):
        if name == "dvclive.monitor_system":
            return fake_monitor
        raise ImportError(name)

    monkeypatch.setattr(dvc_module.importlib, "import_module", _fake_import)

    disabled = dvc_module._disable_dvclive_gpu_monitor_if_unavailable()

    assert disabled is True
    assert fake_monitor.GPU_AVAILABLE is False


def test_disable_dvclive_gpu_monitor_keeps_enabled_when_nvml_works(monkeypatch):
    fake_monitor = types.SimpleNamespace(
        GPU_AVAILABLE=True,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
    )

    def _fake_import(name: str):
        if name == "dvclive.monitor_system":
            return fake_monitor
        raise ImportError(name)

    monkeypatch.setattr(dvc_module.importlib, "import_module", _fake_import)

    disabled = dvc_module._disable_dvclive_gpu_monitor_if_unavailable()

    assert disabled is False
    assert fake_monitor.GPU_AVAILABLE is True


@pytest.mark.skipif(
    shutil.which("dvc") is None,
    reason="dvc executable is required for end-to-end repro validation",
)
def test_dvc_repro_and_force_repro_preserve_non_timing_outputs(tmp_path: Path):
    """End-to-end DVC repro contract for default experiment decomposition.

    Uses DVC-native commands only:
    1) dvc repro
    2) dvc repro --force

    Then validates non-timing output hashes (from dvc.lock) are stable and
    score payloads are identical when timing fields are ignored.
    """

    workspace = tmp_path / "dvc-e2e"
    workspace.mkdir(parents=True, exist_ok=True)

    # DVC stage deps expect these entries in cwd.
    (workspace / "deckard").symlink_to(Path(__file__).resolve().parents[2] / "deckard")
    (workspace / "examples").symlink_to(
        Path(__file__).resolve().parents[2] / "examples",
    )
    rc_path = workspace / ".deckard_rc"
    rc_path.write_text("# test rc\n", encoding="utf-8")

    artifacts_dir = workspace / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    exp = _make_real_experiment_from_examples(artifacts_dir)

    dvc_file = workspace / "dvc.yaml"
    params_file = workspace / "params.yaml"

    pipeline = generate_dvc_pipeline(
        exp,
        output_file=dvc_file.as_posix(),
        params_file=params_file.name,
        mode="single",  # avoid multirun trial-order nondeterminism
        overwrite=True,
    )
    assert "stages" in pipeline and len(pipeline["stages"]) > 0

    # DVC does not accept arbitrary top-level keys in pipeline files.
    dvc_payload = yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}
    if isinstance(dvc_payload, dict):
        dvc_payload.pop("params_payload", None)
    if isinstance(dvc_payload, dict) and "params_file" in dvc_payload:
        dvc_payload.pop("params_file", None)
    if isinstance(dvc_payload, dict):
        stages_payload = dvc_payload.get("stages", {})
        if isinstance(stages_payload, dict):
            for stage_payload in stages_payload.values():
                if isinstance(stage_payload, dict):
                    stage_payload.pop("params", None)
                    cmd = str(stage_payload.get("cmd", "")).strip()
                    if cmd:
                        tokens = cmd.split()
                        filtered = [
                            token
                            for token in tokens
                            if not token.startswith("+params_file=")
                            and not token.startswith("+dvc_file=")
                            and not token.startswith("+dvclive_enabled=")
                            and not token.startswith("dvclive_enabled=")
                        ]
                        filtered.append("dvclive_enabled=false")
                        stage_payload["cmd"] = " ".join(filtered)
    dvc_file.write_text(yaml.safe_dump(dvc_payload, sort_keys=False), encoding="utf-8")

    params_payload = yaml.safe_load(
        (workspace / "examples" / "sklearn" / "config" / "default.yaml").read_text(
            encoding="utf-8",
        ),
    )
    if isinstance(params_payload, dict) and "dvclive_enabled" not in params_payload:
        params_payload["dvclive_enabled"] = False
    params_file.write_text(
        yaml.safe_dump(params_payload, sort_keys=False),
        encoding="utf-8",
    )

    env = _make_runtime_env(rc_path)
    env["DECKARD_CONFIG_DIR"] = (
        workspace / "examples" / "sklearn" / "config"
    ).as_posix()
    env["DECKARD_DEFAULT_CONFIG_FILE"] = "default.yaml"

    init = subprocess.run(
        [sys.executable, "-m", "dvc", "init", "--no-scm", "--quiet"],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert (
        init.returncode == 0
    ), f"dvc init failed:\nSTDOUT:\n{init.stdout}\nSTDERR:\n{init.stderr}"

    repro = subprocess.run(
        [sys.executable, "-m", "dvc", "repro", dvc_file.as_posix()],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
        check=False,
    )
    assert (
        repro.returncode == 0
    ), f"dvc repro failed:\nSTDOUT:\n{repro.stdout}\nSTDERR:\n{repro.stderr}"

    lock_path = workspace / "dvc.lock"
    assert lock_path.exists(), "dvc.lock was not generated by dvc repro"

    def _extract_stage_out_hashes(path: Path) -> dict[str, str]:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        stages = payload.get("stages", {}) if isinstance(payload, dict) else {}
        hashes: dict[str, str] = {}
        for stage_payload in stages.values():
            outs = (
                stage_payload.get("outs", [])
                if isinstance(stage_payload, dict)
                else []
            )
            for out in outs:
                if not isinstance(out, dict):
                    continue
                out_path = str(out.get("path", "")).strip()
                if not out_path:
                    continue
                for key in ("md5", "etag", "checksum", "hash"):
                    if key in out and out.get(key):
                        hashes[out_path] = str(out[key])
                        break
        return hashes

    def _is_volatile_output(path_value: str) -> bool:
        token = str(path_value).lower()
        volatile_parts = (
            "timing",
            "runtime_cache",
            "outputs/logs",
            "run.log",
            "error.log",
            "params.yaml",
        )
        return any(part in token for part in volatile_parts)

    first_hashes_all = _extract_stage_out_hashes(lock_path)
    first_hashes = {
        path: digest
        for path, digest in first_hashes_all.items()
        if (not _is_volatile_output(path)) and ("score" not in path.lower())
    }
    assert (
        len(first_hashes) > 0
    ), "No stable non-timing DVC output hashes were captured"

    score_path = Path(exp.files.score_file)
    assert score_path.exists(), "Expected score file is missing after first repro"
    first_score = json.loads(score_path.read_text(encoding="utf-8"))

    repro_force = subprocess.run(
        [sys.executable, "-m", "dvc", "repro", "--force", dvc_file.as_posix()],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
        check=False,
    )
    assert (
        repro_force.returncode == 0
    ), f"dvc repro --force failed:\nSTDOUT:\n{repro_force.stdout}\nSTDERR:\n{repro_force.stderr}"
    

    second_hashes_all = _extract_stage_out_hashes(lock_path)
    second_hashes = {
        path: digest
        for path, digest in second_hashes_all.items()
        if (not _is_volatile_output(path)) and ("score" not in path.lower())
    }
    assert first_hashes == second_hashes

    second_score = json.loads(score_path.read_text(encoding="utf-8"))

    def _strip_timing_fields(value):
        if isinstance(value, dict):
            out: dict = {}
            for key, item in value.items():
                if "time" in str(key).lower():
                    continue
                out[key] = _strip_timing_fields(item)
            return out
        if isinstance(value, list):
            return [_strip_timing_fields(item) for item in value]
        return value

    first_normalized = _strip_timing_fields(first_score)
    second_normalized = _strip_timing_fields(second_score)
    assert first_normalized == second_normalized
