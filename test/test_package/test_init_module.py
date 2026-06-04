import importlib
import json
import subprocess
import sys

import pytest
from omegaconf import OmegaConf

import deckard


def test_importing_deckard_does_not_register_configs(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "deckard.declarations.register_configs",
        lambda: calls.append("register"),
        raising=True,
    )

    importlib.reload(deckard)

    assert calls == []


def test_warning_policy_is_stable_across_import_orders():
    script_template = """
import importlib
import json
import warnings

modules = {modules!r}
for name in modules:
    importlib.import_module(name)

snapshot = []
for action, message, category, module, lineno in warnings.filters:
    if category.__name__ in {{
        'FutureWarning',
        'DeprecationWarning',
        'UndefinedMetricWarning',
        'RuntimeWarning',
        'ConvergenceWarning',
        'ExperimentalWarning',
        'UserWarning',
    }}:
        snapshot.append((
            action,
            getattr(message, 'pattern', str(message)),
            category.__name__,
            getattr(module, 'pattern', str(module)),
            lineno,
        ))

print(json.dumps(snapshot, sort_keys=True))
"""

    def _run_import_order(modules):
        completed = subprocess.run(
            [sys.executable, "-c", script_template.format(modules=modules)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout.strip())

    first_snapshot = _run_import_order(
        ["deckard", "deckard.experiment.base", "deckard.model.defense.base"],
    )
    second_snapshot = _run_import_order(
        ["deckard.model.defense.base", "deckard.experiment.base", "deckard"],
    )

    assert first_snapshot == second_snapshot


def test_load_yaml_file_reads_content(tmp_path):
    cfg_file = tmp_path / "sample.yaml"
    cfg_file.write_text("root:\n  child: 3\n")

    loaded = deckard._load_yaml_file(cfg_file)

    assert loaded == {"root": {"child": 3}}


def test_file_resolver_validates_argument():
    with pytest.raises(ValueError, match="file resolver requires an argument"):
        deckard._file_resolver("")


def test_file_resolver_raises_for_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(deckard, "DECKARD_CONFIG_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="file resolver: file not found"):
        deckard._file_resolver("does_not_exist.yaml")


def test_file_resolver_reads_nested_key(monkeypatch, tmp_path):
    cfg_file = tmp_path / "nested.yaml"
    cfg_file.write_text("outer:\n  inner:\n    value: ok\n")
    monkeypatch.setattr(deckard, "DECKARD_CONFIG_DIR", str(tmp_path))

    resolved = deckard._file_resolver("nested.yaml:outer.inner")

    assert OmegaConf.to_container(resolved, resolve=True) == {"value": "ok"}


def test_file_resolver_raises_for_missing_key(monkeypatch, tmp_path):
    cfg_file = tmp_path / "nested.yaml"
    cfg_file.write_text("outer:\n  inner: 1\n")
    monkeypatch.setattr(deckard, "DECKARD_CONFIG_DIR", str(tmp_path))

    with pytest.raises(KeyError, match="key 'outer.missing' not found"):
        deckard._file_resolver("nested.yaml:outer.missing")


def test_merge_resolver_merges_fragments_and_resolves_interpolation():
    merged = deckard._merge_resolver(
        {"a": 1, "common": {"x": 1}},
        {"x": 2, "b": "${x}", "common": {"y": 2}},
    )
    as_dict = OmegaConf.to_container(merged, resolve=True)

    assert as_dict["a"] == 1
    assert as_dict["b"] == 2
    assert as_dict["common"] == {"x": 1, "y": 2}


def test_hash_conf_matches_hash_conf_values():
    root = OmegaConf.create({"root": True})
    expected = deckard.hash_conf_values("a", "b", _root_=root)
    observed = deckard._hash_conf("a", "b", _root_=root)
    assert observed == expected


def test_stage_params_excludes_future_stage_components():
    root = OmegaConf.create(
        {
            "stage": "sample",
            "data": {"alias": "adult"},
            "model": {"alias": "rf"},
            "attack": {"alias": "hsj"},
            "score": {"alias": "classification"},
            "directions": ["maximize"],
        },
    )

    payload = deckard._stage_params("???", _root_=root)

    assert payload["stage"] == "sample"
    assert "data" in payload["components"]
    assert "attack" not in payload["components"]
    assert "model" not in payload["components"]


def test_stage_params_includes_attack_for_attack_stage():
    root = OmegaConf.create(
        {
            "stage": "attack",
            "data": {"alias": "adult"},
            "model": {"alias": "rf"},
            "attack": {"alias": "hsj"},
            "detector": {"alias": "spectral"},
            "score": {"alias": "classification"},
        },
    )

    payload = deckard._stage_params("attack", _root_=root)

    assert payload["stage"] == "attack"
    assert "attack" in payload["components"]


def test_stage_params_resolver_accepts_missing_stage_argument():
    root = OmegaConf.create(
        {
            "data": {"alias": "adult"},
            "model": {"alias": "rf"},
        },
    )

    payload = deckard._stage_params("???", _root_=root)

    assert payload["stage"] == "all"


def test_public_api_all_contains_expected_symbols():
    required = {
        "DataConfig",
        "ModelConfig",
        "AttackConfig",
        "DetectorConfig",
        "ExperimentConfig",
        "DefenseConfig",
        "FileConfig",
        "ScorerDictConfig",
    }
    assert required.issubset(set(deckard.__all__))
