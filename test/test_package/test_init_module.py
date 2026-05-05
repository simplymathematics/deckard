import pytest
from omegaconf import OmegaConf

import deckard


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
