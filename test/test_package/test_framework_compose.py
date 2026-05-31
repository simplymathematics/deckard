from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from deckard.model import DefenseConfig


def _compose_from_dir(config_dir: Path, config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name)


def test_sklearn_framework_defense_yaml_composes_in_isolation():
    config_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "sklearn"
        / "config"
        / "frameworks"
        / "sklearn"
    )

    cfg = _compose_from_dir(config_dir, "default_defense")
    assert isinstance(instantiate(cfg), DefenseConfig)


def test_pytorch_framework_defense_yaml_composes_in_isolation():
    pytest.importorskip("torch")

    config_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "pytorch"
        / "config"
        / "frameworks"
        / "pytorch"
    )

    cfg = _compose_from_dir(config_dir, "default_defense")
    assert isinstance(instantiate(cfg), DefenseConfig)


def test_transformers_framework_defense_yaml_composes_in_isolation():
    config_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "transformers"
        / "config"
        / "frameworks"
        / "transformers"
    )

    cfg = _compose_from_dir(config_dir, "default_defense")
    assert isinstance(instantiate(cfg), DefenseConfig)
