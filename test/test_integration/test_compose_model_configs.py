"""Consolidated compose tests for model configs.

This test suite validates that model configuration profiles compose correctly
and produce expected field values. Tests are parametrized to cover representative
model profiles without duplicating test functions.
"""

import pytest
from omegaconf import OmegaConf

from .shared_compose import compose_pytorch, compose_sklearn


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
    cfg = compose_sklearn(config_name)
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
    cfg = compose_pytorch(config_name)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    for field_name, expected_value in expected_fields.items():
        assert model_cfg[field_name] == expected_value


def test_sklearn_default_can_override_model_profile():
    """Test that default config can be overridden with model override."""
    cfg = compose_sklearn("default", overrides=["model=test-logistic"])
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["name"] == "sklearn.linear_model.LogisticRegression"
    assert model_cfg["classifier"] is True
    assert model_cfg["alias"] == "test_logistic"
