"""Consolidated compose tests for score configs.

This test suite validates that score configuration profiles compose correctly
and produce expected field values. Tests are parametrized to cover representative
score profiles without duplicating test functions.
"""

import pytest
from omegaconf import OmegaConf

from .shared_compose import compose_sklearn


@pytest.mark.parametrize(
    "config_name,expected_scorers",
    [
        pytest.param(
            "default",
            ["accuracy", "precision", "recall", "f1"],
            id="sklearn-classification",
            marks=pytest.mark.parametrize(
                "overrides",
                [["score=classification"]],
                indirect=False,
            ),
        ),
    ],
)
def test_sklearn_score_config_composes(
    config_name: str,
    expected_scorers: list[str],
):
    """Test sklearn score config profiles compose correctly."""
    cfg = compose_sklearn(config_name, overrides=["score=classification"])
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    for scorer in expected_scorers:
        assert scorer in score_cfg["scorers"]


def test_sklearn_survival_score_group_composes():
    """Test sklearn survival score group composes with survival scorers."""
    cfg = compose_sklearn("survival")
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    assert "concordance" in score_cfg["scorers"]
    assert "aic" in score_cfg["scorers"]
    assert "bic" in score_cfg["scorers"]
