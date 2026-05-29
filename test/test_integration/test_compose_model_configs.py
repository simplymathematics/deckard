"""Consolidated compose tests for model configs.

This test suite validates that model configuration profiles compose correctly
and produce expected field values. Tests are parametrized to cover representative
model profiles without duplicating test functions.
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
            "model/logistic",
            {
                "name": "sklearn.linear_model.LogisticRegression",
                "classifier": True,
                "alias": "logistic",
            },
            id="sklearn-logistic",
        ),
        pytest.param(
            "model/cox",
            {
                "name": "lifelines.fitters.coxph_fitter.CoxPHFitter",
                "classifier": False,
                "alias": "cox",
            },
            id="sklearn-cox",
        ),
    ],
)
def test_sklearn_model_config_composes(config_name: str, expected_fields: dict):
    """Test sklearn model config profiles compose and contain expected fields."""
    cfg = _compose_sklearn(config_name)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    for field_name, expected_value in expected_fields.items():
        assert model_cfg[field_name] == expected_value


@pytest.mark.parametrize(
    "config_name,expected_fields",
    [
        pytest.param(
            "model/tinynet",
            {
                "name": "deckard.frameworks.pytorch.model.TinyNet",
                "classifier": True,
                "alias": "tinynet",
            },
            id="pytorch-tinynet",
        ),
    ],
)
def test_pytorch_model_config_composes(config_name: str, expected_fields: dict):
    """Test pytorch model config profiles compose and contain expected fields."""
    cfg = _compose_pytorch(config_name)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    for field_name, expected_value in expected_fields.items():
        assert model_cfg[field_name] == expected_value


def test_sklearn_default_can_override_model_profile():
    """Test that default config can be overridden with model override."""
    cfg = _compose_sklearn("default", overrides=["model=test-logistic"])
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["name"] == "sklearn.linear_model.LogisticRegression"
    assert model_cfg["classifier"] is True
    assert model_cfg["alias"] == "test_logistic"
