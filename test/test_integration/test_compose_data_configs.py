"""Consolidated compose tests for data configs.

This test suite validates that data configuration profiles compose correctly
and produce expected field values. Tests are parametrized from actual
``examples/*/config/data/*.yaml`` files to ensure coverage stays in sync with
the repository configuration surface.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from .shared_compose import (
    PYTORCH_CONFIG_DIR,
    SKLEARN_CONFIG_DIR,
    compose_pytorch,
    compose_sklearn,
)


def _discover_data_profile_names(config_dir: Path) -> list[str]:
    """Return sorted ``data/<name>`` config names from ``config/data/*.yaml``."""
    data_dir = config_dir / "data"
    return sorted(
        f"data/{path.stem}" for path in data_dir.glob("*.yaml") if path.is_file()
    )


SKLEARN_DATA_CONFIG_NAMES = _discover_data_profile_names(SKLEARN_CONFIG_DIR)
PYTORCH_DATA_CONFIG_NAMES = _discover_data_profile_names(PYTORCH_CONFIG_DIR)


@pytest.mark.parametrize("config_name", SKLEARN_DATA_CONFIG_NAMES)
def test_sklearn_data_config_composes(config_name: str):
    """Test all sklearn data config profiles compose and expose core fields."""
    cfg = compose_sklearn(config_name)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)

    assert isinstance(data_cfg.get("name"), str)
    assert data_cfg.get("name", "").strip() != ""


@pytest.mark.parametrize("config_name", PYTORCH_DATA_CONFIG_NAMES)
def test_pytorch_data_config_composes(config_name: str):
    """Test all pytorch data config profiles compose and expose core fields."""
    cfg = compose_pytorch(config_name)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_cfg, dict)
    assert isinstance(data_cfg.get("name"), str)
    assert data_cfg.get("name", "").strip() != ""
