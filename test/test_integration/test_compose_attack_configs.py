"""Consolidated compose tests for attack configs.

This test suite validates that attack configuration profiles are correctly
included in default composed configurations.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra

SKLEARN_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
PYTORCH_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "config"


def _reset_hydra_state():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_store = ConfigStore.instance()
    for key in list(config_store.repo.keys()):
        if key not in {"hydra", "_dummy_empty_config_.yaml"}:
            config_store.repo.pop(key, None)


def _compose_sklearn(config_name: str, overrides: list[str] | None = None):
    """Compose config from sklearn config directory."""
    overrides = overrides or []
    _reset_hydra_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(SKLEARN_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def _compose_pytorch(config_name: str, overrides: list[str] | None = None):
    """Compose config from pytorch config directory."""
    overrides = overrides or []
    _reset_hydra_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(PYTORCH_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def test_sklearn_default_includes_attack_config():
    """Test that sklearn default config includes attack configuration."""
    cfg = _compose_sklearn("default")
    assert cfg is not None
    assert "attack" in cfg
    # Default attack is hsj
    assert cfg.attack.alias == "hsj"


def test_pytorch_torch_default_includes_attack_config():
    """Test that pytorch torch_default config includes attack configuration."""
    cfg = _compose_pytorch("torch_default")
    assert cfg is not None
    assert "attack" in cfg
    # Default attack is fgm
    assert cfg.attack.alias == "fgm"


