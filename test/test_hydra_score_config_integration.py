from pathlib import Path
import importlib.util
import math
import tempfile

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deckard.score import ScorerDictConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "sklearn" / "config"


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


# ---------------------------------------------------------------------------
# Hash stability and persistence
# ---------------------------------------------------------------------------

def test_scorer_dict_config_hash_stable_after_scoring():
    cfg = _compose("default", overrides=["score=classification"])
    scorer = ScorerDictConfig(**OmegaConf.to_container(cfg.score, resolve=True))
    original_hash = hash(scorer)
    scorer(y_true=[1, 0, 1, 1], y_pred=[1, 0, 0, 1], mode=None)
    scorer.score_dict["extra"] = 42
    assert hash(scorer) == original_hash


def test_scorer_dict_config_equal_content_produces_equal_hash():
    cfg = _compose("default", overrides=["score=classification"])
    raw = OmegaConf.to_container(cfg.score, resolve=True)
    scorer_a = ScorerDictConfig(**raw)
    scorer_b = ScorerDictConfig(**raw)
    assert hash(scorer_a) == hash(scorer_b)


def test_scorer_dict_config_scores_persist_and_reload():
    cfg = _compose("default", overrides=["score=classification"])
    scorer = ScorerDictConfig(**OmegaConf.to_container(cfg.score, resolve=True))
    scores = scorer(y_true=[1, 0, 1, 1], y_pred=[1, 0, 0, 1], mode=None)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "scorer_scores.json"
        scorer.save_scores(scores, path)
        loaded = scorer.load_scores(str(path))
    assert "accuracy" in loaded


def test_scorer_dict_config_object_pickle_roundtrip():
    cfg = _compose("default", overrides=["score=classification"])
    scorer = ScorerDictConfig(**OmegaConf.to_container(cfg.score, resolve=True))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "scorer.pkl"
        scorer.save_object(scorer, str(path))
        loaded = scorer.load_object(str(path))
    assert isinstance(loaded, ScorerDictConfig)
