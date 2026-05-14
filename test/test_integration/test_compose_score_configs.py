"""Consolidated compose tests for score configs.

This test suite validates that score configuration profiles compose correctly
and produce expected field values. Tests are parametrized to cover representative
score profiles without duplicating test functions.
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
    "config_name,expected_scorers",
    [
        pytest.param(
            "default",
            ["accuracy", "precision", "recall", "f1"],
            id="sklearn-classification",
            marks=pytest.mark.parametrize(
                "overrides", [["score=classification"]], indirect=False
            ),
        ),
    ],
)
def test_sklearn_score_config_composes(
    config_name: str, expected_scorers: list[str]
):
    """Test sklearn score config profiles compose correctly."""
    cfg = _compose_sklearn(config_name, overrides=["score=classification"])
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    for scorer in expected_scorers:
        assert scorer in score_cfg["scorers"]


def test_sklearn_survival_score_group_composes():
    """Test sklearn survival score group composes with survival scorers."""
    cfg = _compose_sklearn("survival")
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    assert "concordance" in score_cfg["scorers"]
    assert "aic" in score_cfg["scorers"]
    assert "bic" in score_cfg["scorers"]
