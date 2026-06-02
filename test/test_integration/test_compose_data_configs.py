"""Consolidated compose tests for data configs.

This test suite validates that data configuration profiles compose correctly
and produce expected field values. Tests are parametrized from actual
``examples/*/config/data/*.yaml`` files to ensure coverage stays in sync with
the repository configuration surface.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

SKLEARN_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
)
PYTORCH_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "config"
)


def _discover_data_profile_names(config_dir: Path) -> list[str]:
    """Return sorted ``data/<name>`` config names from ``config/data/*.yaml``."""
    data_dir = config_dir / "data"
    return sorted(
        f"data/{path.stem}" for path in data_dir.glob("*.yaml") if path.is_file()
    )


SKLEARN_DATA_CONFIG_NAMES = _discover_data_profile_names(SKLEARN_CONFIG_DIR)
PYTORCH_DATA_CONFIG_NAMES = _discover_data_profile_names(PYTORCH_CONFIG_DIR)


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


@pytest.mark.parametrize("config_name", SKLEARN_DATA_CONFIG_NAMES)
def test_sklearn_data_config_composes(config_name: str):
    """Test all sklearn data config profiles compose and expose core fields."""
    cfg = _compose_sklearn(config_name)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)

    assert isinstance(data_cfg.get("name"), str)
    assert data_cfg.get("name", "").strip() != ""


@pytest.mark.parametrize("config_name", PYTORCH_DATA_CONFIG_NAMES)
def test_pytorch_data_config_composes(config_name: str):
    """Test all pytorch data config profiles compose and expose core fields."""
    cfg = _compose_pytorch(config_name)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)
    assert isinstance(data_cfg.get("name"), str)
    assert data_cfg.get("name", "").strip() != ""
