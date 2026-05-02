from pathlib import Path
import importlib.util
import math

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deckard.score import ScorerDictConfig

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _compose(config_name: str, overrides: list[str] | None = None):
    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def test_survival_config_uses_survival_score_group():
    cfg = _compose("survival")
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    assert "concordance" in score_cfg["scorers"]
    assert "aic" in score_cfg["scorers"]
    assert "bic" in score_cfg["scorers"]


def test_classification_score_group_executes_end_to_end():
    cfg = _compose("default", overrides=["score=classification"])
    scorer = ScorerDictConfig(**OmegaConf.to_container(cfg.score, resolve=True))

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    scores = scorer(y_true=y_true, y_pred=y_pred, mode=None)

    assert "accuracy" in scores
    assert "precision" in scores
    assert "recall" in scores
    assert "f1" in scores
    assert "log_loss" in scores


def test_regression_score_group_executes_end_to_end():
    cfg = _compose("default", overrides=["score=regression"])
    scorer = ScorerDictConfig(**OmegaConf.to_container(cfg.score, resolve=True))

    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.2, 3.8]
    scores = scorer(y_true=y_true, y_pred=y_pred, mode=None)

    assert "mse" in scores
    assert "rmse" in scores
    assert "mae" in scores
    assert "r2" in scores


def test_survival_score_group_executes_end_to_end():
    cfg = _compose("survival")
    scorer = ScorerDictConfig(**OmegaConf.to_container(cfg.score, resolve=True))

    class _MockFitter:
        concordance_index_ = 0.73
        log_likelihood_ = -52.0
        params_ = [1.0, 2.0, 3.0]

    scores = scorer(y_true=[1, 2, 3, 4], y_pred=_MockFitter(), mode=None)
    assert "concordance" in scores
    assert "aic" in scores
    assert "bic" in scores
    assert math.isfinite(scores["aic"])
    assert math.isfinite(scores["bic"])


@pytest.mark.skipif(
    importlib.util.find_spec("fairlearn") is None,
    reason="fairlearn is required to validate fairness score profile integration",
)
def test_default_can_switch_to_fairness_score_group():
    cfg = _compose("default", overrides=["data=fair-adult", "score=fairness-classification"])
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    assert "demographic_parity_difference" in score_cfg["scorers"]
    assert "equalized_odds_difference" in score_cfg["scorers"]


@pytest.mark.skipif(
    importlib.util.find_spec("fairlearn") is None,
    reason="fairlearn is required to validate fairness score runtime",
)
def test_fairness_score_group_executes_end_to_end():
    cfg = _compose("default", overrides=["data=fair-adult", "score=fairness-classification"])
    scorer = ScorerDictConfig(**OmegaConf.to_container(cfg.score, resolve=True))

    y_true = [1, 0, 1, 0]
    y_pred = [1, 1, 1, 0]
    sensitive_features = [0, 0, 1, 1]

    scores = scorer(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        mode=None,
    )

    assert "demographic_parity_difference" in scores
    assert "equalized_odds_difference" in scores
