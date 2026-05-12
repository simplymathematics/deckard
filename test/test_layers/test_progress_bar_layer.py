from pathlib import Path
from datetime import datetime

import optuna
import pytest
import yaml

from deckard.layers import progress_bar as progress_bar_module


def _write_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


class DummyPbar:
    def __init__(self, total, initial, desc, unit, position):
        self.total = total
        self.n = initial
        self.desc = desc
        self.unit = unit
        self.position = position
        self.start_t = None
        self.last_print_t = None
        self.closed = False

    def update(self, amount):
        self.n += amount

    def close(self):
        self.closed = True


def test_count_studies_for_stage_conf_variants():
    assert (
        progress_bar_module._count_studies_for_stage_conf(
            {"matrix": {"a": [1, 2], "b": ["x", "y", "z"]}},
        )
        == 6
    )
    assert (
        progress_bar_module._count_studies_for_stage_conf(
            {"foreach": ["a", "b", "c"]},
        )
        == 3
    )
    assert (
        progress_bar_module._count_studies_for_stage_conf(
            {"cmd": "deckard optimize --multirun"},
        )
        == 1
    )


def test_infer_stages_from_dvc_selects_multirun_only(tmp_path):
    dvc_file = tmp_path / "dvc.yaml"
    _write_yaml(
        dvc_file,
        {
            "stages": {
                "train": {"cmd": "deckard optimize --multirun"},
                "eval": {"cmd": "deckard score"},
                "attack": {"cmd": ["deckard", "optimize", "--multirun"]},
            },
        },
    )

    stages = progress_bar_module._infer_stages_from_dvc(str(dvc_file))

    assert stages == ["train", "attack"]


def test_extract_and_resolve_stage_config_name(tmp_path):
    cfg_dir = tmp_path / "config"
    default_cfg = cfg_dir / "default.yaml"
    stage_cfg = cfg_dir / "inference-default.yaml"
    _write_yaml(
        default_cfg,
        {
            "hydra": {
                "sweeper": {"storage": "sqlite:///db.sqlite3", "n_trials": 1},
            },
        },
    )
    _write_yaml(
        stage_cfg,
        {
            "hydra": {
                "sweeper": {"storage": "sqlite:///db.sqlite3", "n_trials": 5},
            },
        },
    )

    stage_conf = {
        "cmd": "deckard optimize --multirun --config-name inference-default",
    }

    assert (
        progress_bar_module._extract_stage_config_name(stage_conf)
        == "inference-default"
    )
    resolved = progress_bar_module._resolve_hydra_config_for_stage(
        stage_conf,
        str(default_cfg),
    )
    assert Path(resolved) == stage_cfg


def test_get_hydra_sweeper_config_supports_flat_or_nested_hydra(tmp_path):
    nested_cfg = tmp_path / "nested.yaml"
    flat_cfg = tmp_path / "flat.yaml"

    _write_yaml(
        nested_cfg,
        {
            "hydra": {
                "sweeper": {
                    "storage": "sqlite:///nested.sqlite3",
                    "n_trials": 11,
                },
            },
        },
    )
    _write_yaml(flat_cfg, {"sweeper": {"storage": "sqlite:///flat.sqlite3"}})

    assert progress_bar_module._get_hydra_sweeper_config(str(nested_cfg)) == (
        "sqlite:///nested.sqlite3",
        11,
    )
    # Defaults to 100 when n_trials is not set.
    assert progress_bar_module._get_hydra_sweeper_config(str(flat_cfg)) == (
        "sqlite:///flat.sqlite3",
        100,
    )


def test_collect_storage_finished_counts_reads_trials_and_earliest(tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'optuna.sqlite3').as_posix()}"

    for study_name, n_trials in (("s1", 2), ("s2", 1)):
        study = optuna.create_study(
            study_name=study_name,
            storage=db_url,
            load_if_exists=True,
        )
        study.optimize(lambda trial: 1.0, n_trials=n_trials)

    counts, earliest = progress_bar_module._collect_storage_finished_counts(
        db_url,
        {"COMPLETE", "FAILED", "PRUNED"},
    )

    assert sorted(counts) == [1, 2]
    assert earliest is not None


def test_progress_bar_main_integration_reads_dvc_and_optuna_db(
    tmp_path,
    monkeypatch,
):
    cfg_dir = tmp_path / "config"
    dvc_file = tmp_path / "dvc.yaml"
    default_cfg = cfg_dir / "default.yaml"
    stage_cfg = cfg_dir / "stage-a.yaml"
    db_url = f"sqlite:///{(tmp_path / 'optuna.sqlite3').as_posix()}"

    # Expected studies = len(foreach)=2 and expected trials = 2*2=4.
    _write_yaml(
        dvc_file,
        {
            "stages": {
                "opt_stage": {
                    "foreach": ["left", "right"],
                    "cmd": "deckard optimize --multirun --config-name stage-a",
                },
            },
        },
    )
    _write_yaml(
        default_cfg,
        {"hydra": {"sweeper": {"storage": db_url, "n_trials": 2}}},
    )
    _write_yaml(
        stage_cfg,
        {"hydra": {"sweeper": {"storage": db_url, "n_trials": 2}}},
    )

    for study_name in ("study_left", "study_right"):
        study = optuna.create_study(
            study_name=study_name,
            storage=db_url,
            load_if_exists=True,
        )
        study.optimize(lambda trial: 0.1, n_trials=2)

    monkeypatch.setattr(progress_bar_module, "tqdm", DummyPbar)

    result = progress_bar_module.progress_bar_main(
        hydra_cfg_file=str(default_cfg),
        dvc_file=str(dvc_file),
        poll_interval=0.0,
    )

    assert result["storages"] == [db_url]
    assert result["expected_studies"] == 2
    assert result["expected_trials"] == 4
    assert result["completed_studies"] == 2
    assert result["completed_trials"] == 4
    assert result["start_time"] is not None


def test_progress_bar_main_raises_for_unknown_stage(tmp_path):
    dvc_file = tmp_path / "dvc.yaml"
    default_cfg = tmp_path / "config" / "default.yaml"
    _write_yaml(
        dvc_file,
        {"stages": {"known": {"cmd": "deckard optimize --multirun"}}},
    )
    _write_yaml(
        default_cfg,
        {
            "hydra": {
                "sweeper": {"storage": "sqlite:///x.sqlite3", "n_trials": 1},
            },
        },
    )

    with pytest.raises(KeyError, match="Stage 'missing' was not found"):
        progress_bar_module.progress_bar_main(
            hydra_cfg_file=str(default_cfg),
            dvc_file=str(dvc_file),
            stages="missing",
            poll_interval=0.0,
        )


def test_calculate_number_of_studies_per_stage_and_missing_stage(tmp_path):
    dvc_file = tmp_path / "dvc.yaml"
    _write_yaml(
        dvc_file,
        {
            "stages": {
                "mx": {"matrix": {"a": [1, 2], "b": [3, 4]}},
            },
        },
    )

    assert (
        progress_bar_module.calculate_number_of_studies_per_stage("mx", str(dvc_file))
        == 4
    )
    with pytest.raises(KeyError, match="Stage 'missing' was not found"):
        progress_bar_module.calculate_number_of_studies_per_stage(
            "missing",
            str(dvc_file),
        )


def test_infer_stages_from_dvc_raises_when_no_multirun(tmp_path):
    dvc_file = tmp_path / "dvc.yaml"
    _write_yaml(dvc_file, {"stages": {"single": {"cmd": "deckard score"}}})

    with pytest.raises(ValueError, match="No multirun stages found"):
        progress_bar_module._infer_stages_from_dvc(str(dvc_file))


def test_extract_stage_config_name_supports_equals_form_and_empty():
    assert (
        progress_bar_module._extract_stage_config_name(
            {"cmd": "deckard optimize --config-name=abc"},
        )
        == "abc"
    )
    assert progress_bar_module._extract_stage_config_name({"cmd": "   "}) is None


def test_resolve_hydra_config_for_stage_absolute_and_not_found(tmp_path):
    default_cfg = tmp_path / "default.yaml"
    _write_yaml(default_cfg, {"hydra": {"sweeper": {"storage": "sqlite:///x.db"}}})

    abs_cfg = tmp_path / "absolute.yaml"
    _write_yaml(abs_cfg, {"hydra": {"sweeper": {"storage": "sqlite:///abs.db"}}})

    resolved = progress_bar_module._resolve_hydra_config_for_stage(
        {"cmd": f"deckard optimize --config-name {abs_cfg.as_posix()}"},
        str(default_cfg),
    )
    assert resolved == abs_cfg.as_posix()

    with pytest.raises(FileNotFoundError, match="Could not resolve config"):
        progress_bar_module._resolve_hydra_config_for_stage(
            {"cmd": "deckard optimize --config-name missing_cfg"},
            str(default_cfg),
        )


def test_parser_helpers_cover_nested_csv_and_ranges():
    assert progress_bar_module._split_top_level_commas("a,b,c") == ["a", "b", "c"]
    assert progress_bar_module._split_top_level_commas(
        'choice("a,b",range(0,2),x)'
    ) == [
        'choice("a,b",range(0,2),x)',
    ]
    assert progress_bar_module._unwrap_outer_call("range(1,5)", "range") == "1,5"
    assert progress_bar_module._unwrap_outer_call("value", "range") is None
    assert progress_bar_module._safe_float("1.5") == 1.5
    assert progress_bar_module._safe_float("not-a-float") is None

    assert progress_bar_module._count_values_for_param_space([1, 2, 3]) == 3
    assert progress_bar_module._count_values_for_param_space(10) == 1
    assert progress_bar_module._count_values_for_param_space(None) is None
    assert progress_bar_module._count_values_for_param_space("") is None
    assert progress_bar_module._count_values_for_param_space("choice(a,b,c)") == 3
    assert progress_bar_module._count_values_for_param_space("int(range(1,4))") == 3
    assert progress_bar_module._count_values_for_param_space("range(5,1,1)") == 0
    assert progress_bar_module._count_values_for_param_space("range(1,5,0)") is None
    assert progress_bar_module._count_values_for_param_space("x,y,z") == 3

    assert (
        progress_bar_module._calculate_grid_search_n_trials({"a": [1, 2], "b": "x,y"})
        == 4
    )
    assert progress_bar_module._calculate_grid_search_n_trials("not-a-dict") is None


def test_get_hydra_sweeper_config_grid_sampler_and_errors(tmp_path):
    grid_cfg = tmp_path / "grid.yaml"
    _write_yaml(
        grid_cfg,
        {
            "hydra": {
                "sweeper": {
                    "storage": "sqlite:///grid.sqlite3",
                    "n_trials": 50,
                    "sampler": {
                        "_target_": "optuna.samplers.GridSampler",
                    },
                    "params": {
                        "model.alpha": "range(0,3)",
                        "data.seed": "choice(1,2)",
                    },
                },
            },
        },
    )
    storage, n_trials = progress_bar_module._get_hydra_sweeper_config(str(grid_cfg))
    assert storage == "sqlite:///grid.sqlite3"
    assert n_trials == 6

    invalid_root_cfg = tmp_path / "invalid.yaml"
    invalid_root_cfg.write_text("- one\n- two\n")
    with pytest.raises(ValueError, match="Expected mapping at root"):
        progress_bar_module._get_hydra_sweeper_config(str(invalid_root_cfg))

    missing_storage_cfg = tmp_path / "missing_storage.yaml"
    _write_yaml(missing_storage_cfg, {"hydra": {"sweeper": {"n_trials": 2}}})
    with pytest.raises(ValueError, match="No hydra.sweeper.storage"):
        progress_bar_module._get_hydra_sweeper_config(str(missing_storage_cfg))


def test_progress_bar_main_polls_until_completion(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "config"
    dvc_file = tmp_path / "dvc.yaml"
    default_cfg = cfg_dir / "default.yaml"
    _write_yaml(
        dvc_file,
        {"stages": {"stage": {"cmd": "deckard optimize --multirun"}}},
    )
    _write_yaml(
        default_cfg,
        {"hydra": {"sweeper": {"storage": "sqlite:///poll.sqlite3", "n_trials": 2}}},
    )

    calls = iter(
        [
            ([0], None),
            ([2], datetime(2026, 1, 1, 0, 0, 0)),
        ],
    )

    def fake_collect(*_args, **_kwargs):
        try:
            return next(calls)
        except StopIteration:
            return ([2], datetime(2026, 1, 1, 0, 0, 0))

    monkeypatch.setattr(
        progress_bar_module, "_collect_storage_finished_counts", fake_collect
    )
    monkeypatch.setattr(progress_bar_module, "tqdm", DummyPbar)
    monkeypatch.setattr(progress_bar_module.time, "sleep", lambda *_a, **_k: None)

    result = progress_bar_module.progress_bar_main(
        hydra_cfg_file=str(default_cfg),
        dvc_file=str(dvc_file),
        stages="stage",
        poll_interval=0.0,
    )

    assert result["expected_studies"] == 1
    assert result["expected_trials"] == 2
    assert result["completed_studies"] == 1
    assert result["completed_trials"] == 2
