"""Consolidated compose tests for data configs.

This test suite validates that data configuration profiles compose correctly
and produce expected field values. Tests are parametrized to cover representative
data profiles without duplicating test functions.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

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


@pytest.mark.parametrize(
    "config_name,expected_fields",
    [
        pytest.param(
            "data/adult",
            {"dataset_name": "adult", "alias": "adult"},
            id="sklearn-adult",
        ),
        pytest.param(
            "data/anjana",
            {
                "dataset_name": "make_classification",
                "_target_": "deckard.plugins.anjana.AnjanaDataConfig",
                "alias": "anjana",
            },
            id="sklearn-anjana",
        ),
        pytest.param(
            "data/fair-adult",
            {
                "dataset_name": "adult",
                "_target_": "deckard.plugins.fairlearn.FairlearnDataConfig",
                "sensitive_columns": ["sex"],
            },
            id="sklearn-fairlearn",
        ),
        pytest.param(
            "data/lung",
            {"dataset_name": "lung", "target": "E", "classifier": False},
            id="sklearn-lifelines",
        ),
    ],
)
def test_sklearn_data_config_composes(config_name: str, expected_fields: dict):
    """Test sklearn data config profiles compose and contain expected fields."""
    cfg = _compose_sklearn(config_name)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    for field_name, expected_value in expected_fields.items():
        assert data_cfg[field_name] == expected_value


@pytest.mark.parametrize(
    "config_name,expected_fields",
    [
        pytest.param(
            "data/torch_mnist",
            {"dataset_name": "torch_mnist", "alias": "torch_mnist"},
            id="pytorch-mnist",
        ),
    ],
)
def test_pytorch_data_config_composes(config_name: str, expected_fields: dict):
    """Test pytorch data config profiles compose and contain expected fields."""
    cfg = _compose_pytorch(config_name)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    for field_name, expected_value in expected_fields.items():
        assert data_cfg[field_name] == expected_value
