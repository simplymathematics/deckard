from types import SimpleNamespace

import pytest

from deckard.layers import rerun_failed_studies as mod


def test_load_meta_schema_success_and_errors(tmp_path):
    schema_file = tmp_path / "meta.yaml"
    schema_file.write_text("schema:\n  sep: '_'\n  data: 0\n", encoding="utf-8")

    schema = mod._load_meta_schema(str(schema_file))
    assert schema["sep"] == "_"
    assert schema["data"] == 0

    with pytest.raises(FileNotFoundError):
        mod._load_meta_schema(str(tmp_path / "missing.yaml"))

    bad_schema = tmp_path / "bad.yaml"
    bad_schema.write_text("schema: [1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a dictionary"):
        mod._load_meta_schema(str(bad_schema))


def test_parse_study_metadata_supports_int_and_range_locators():
    schema = {"sep": "_", "data": 0, "model": 1, "attack": "2:3"}
    parsed = mod._parse_study_metadata("adult_weibull_mia_v1", schema)

    assert parsed["data"] == "adult"
    assert parsed["model"] == "weibull"
    assert parsed["attack"] == "mia_v1"

    with pytest.raises(ValueError, match="Unsupported schema locator"):
        mod._parse_study_metadata("adult_weibull", {"sep": "_", "data": object()})


def test_build_rerun_command_for_study_returns_none_when_no_overrides():
    schema = {"sep": "_", "ignored": 5}
    assert mod._build_rerun_command_for_study("short_name", schema=schema) is None

    cmd = mod._build_rerun_command_for_study(
        "adult_weibull_membership",
        schema={"sep": "_", "data": 0, "model": 1, "attack": 2},
    )
    assert cmd is not None
    assert "deckard optimize" in cmd
    assert "data=adult" in cmd
    assert "model=weibull" in cmd
    assert "attack=membership" in cmd


def test_collect_failed_studies_include_running_toggle(monkeypatch):
    class DummyTrial:
        def __init__(self, state):
            self.state = state

    class DummyStudy:
        def __init__(self, states):
            self._states = states

        def get_trials(self, deepcopy=False):
            return [DummyTrial(state) for state in self._states]

    summaries = [
        SimpleNamespace(study_name="only_fail"),
        SimpleNamespace(study_name="fail_and_running"),
        SimpleNamespace(study_name="has_complete"),
        SimpleNamespace(name="fallback_name"),
        SimpleNamespace(study_name=""),
    ]

    studies = {
        "only_fail": DummyStudy(["TrialState.FAIL"]),
        "fail_and_running": DummyStudy(["TrialState.FAIL", "TrialState.RUNNING"]),
        "has_complete": DummyStudy(["TrialState.FAIL", "TrialState.COMPLETE"]),
        "fallback_name": DummyStudy(["TrialState.FAIL"]),
    }

    monkeypatch.setattr(
        mod.optuna.study,
        "get_all_study_summaries",
        lambda storage: summaries,
    )
    monkeypatch.setattr(
        mod.optuna.study,
        "load_study",
        lambda storage, study_name: studies[study_name],
    )

    failed_default = mod._collect_failed_studies(
        "sqlite:///optuna.db", include_running=False
    )
    assert failed_default == ["only_fail", "fallback_name"]

    failed_with_running = mod._collect_failed_studies(
        "sqlite:///optuna.db", include_running=True
    )
    assert failed_with_running == ["only_fail", "fail_and_running", "fallback_name"]


def test_rerun_failed_studies_main_dry_run_and_execute(monkeypatch, tmp_path):
    meta_file = tmp_path / "meta.yaml"
    meta_file.write_text(
        "schema:\n  sep: '_'\n  data: 0\n  model: 1\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        mod,
        "_collect_failed_studies",
        lambda storage, include_running=False: ["adult_weibull", "___"],
    )

    dry_run = mod.rerun_failed_studies_main(
        optuna_db="sqlite:///optuna.db",
        working_dir=str(tmp_path),
        meta_schema=str(meta_file),
        execute=False,
        limit=1,
    )
    assert dry_run["executed"] is False
    assert dry_run["failed_studies"] == ["adult_weibull"]
    assert len(dry_run["planned"]) == 1
    assert dry_run["results"] == []

    def fake_run(command, cwd, capture_output, text):
        assert command[0] == "bash"
        assert cwd == str(tmp_path)
        assert capture_output is True
        assert text is True
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    execute = mod.rerun_failed_studies_main(
        optuna_db="sqlite:///optuna.db",
        working_dir=str(tmp_path),
        meta_schema=str(meta_file),
        execute=True,
        limit=None,
    )

    assert execute["executed"] is True
    assert len(execute["planned"]) == 1
    assert execute["skipped"] == ["___"]
    assert execute["results"][0]["returncode"] == 0


def test_rerun_failed_studies_parser_flags_and_defaults():
    args = mod.rerun_failed_studies_parser.parse_args(
        ["--include_running", "--execute", "--limit", "7"]
    )
    assert args.optuna_db == "sqlite:///optuna.db"
    assert args.working_dir == "."
    assert args.meta_schema == "config/meta.yaml"
    assert args.include_running is True
    assert args.execute is True
    assert args.limit == 7
