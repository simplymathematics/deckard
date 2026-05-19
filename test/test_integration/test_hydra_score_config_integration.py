import importlib.util
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import optuna
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

os.environ.setdefault("DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION", "1")

from helpers import make_runtime_env, reset_hydra_state

from deckard.score import ScorerDictConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
EXAMPLES_SKLEARN_DIR = Path(__file__).resolve().parents[2] / "examples" / "sklearn"
DECKARD_RC_PATH = EXAMPLES_SKLEARN_DIR / ".deckard_rc"

# Representative (non-exhaustive) compose/execution combo used for score validation.
# This intentionally avoids sweeping across all possible config cross-products.
SELECTED_SCORE_OVERRIDES = [
    "data=test-classification",
    "model=test-logistic",
    "attack=hsj",
    "defense=class-labels",
    "score=classification",
]


def _compose(config_name: str, overrides: list[str] | None = None):
    overrides = overrides or []
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_store = ConfigStore.instance()
    for key in list(config_store.repo.keys()):
        if key not in {"hydra", "_dummy_empty_config_.yaml"}:
            config_store.repo.pop(key, None)
    reset_hydra_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def _score_kwargs(cfg):
    score_kwargs = OmegaConf.to_container(cfg.score.scorers, resolve=True)
    return score_kwargs


def test_survival_config_uses_survival_score_group():
    cfg = _compose("survival")
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    assert "concordance" in score_cfg["scorers"]
    assert "aic" in score_cfg["scorers"]
    assert "bic" in score_cfg["scorers"]


def test_classification_score_group_executes_end_to_end():
    cfg = _compose("default", overrides=["score=classification"])
    scorer = ScorerDictConfig(scorers=_score_kwargs(cfg))

    y_true = [1, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    y_proba = [0.9, 0.1, 0.3, 0.8]
    scores = scorer(y_true=y_true, y_pred=y_pred, y_proba=y_proba, mode=None)

    assert "accuracy" in scores
    assert "precision" in scores
    assert "recall" in scores
    assert "f1" in scores
    assert "log_loss" in scores


def test_regression_score_group_executes_end_to_end():
    cfg = _compose("default", overrides=["score=regression"])
    scorer = ScorerDictConfig(scorers=_score_kwargs(cfg))

    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 1.9, 3.2, 3.8]
    scores = scorer(y_true=y_true, y_pred=y_pred, mode=None)

    assert "mse" in scores
    assert "rmse" in scores
    assert "mae" in scores
    assert "r2" in scores


def test_survival_score_group_executes_end_to_end():
    cfg = _compose("survival")
    scorer = ScorerDictConfig(scorers=_score_kwargs(cfg))

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
    cfg = _compose(
        "default",
        overrides=["data=fair-adult", "score=fairness-classification"],
    )
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    assert "demographic_parity_difference" in score_cfg["scorers"]
    assert "equalized_odds_difference" in score_cfg["scorers"]


@pytest.mark.skipif(
    importlib.util.find_spec("fairlearn") is None,
    reason="fairlearn is required to validate fairness score runtime",
)
def test_fairness_score_group_executes_end_to_end():
    cfg = _compose(
        "default",
        overrides=["data=fair-adult", "score=fairness-classification"],
    )
    scorer = ScorerDictConfig(scorers=_score_kwargs(cfg))

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
    scorer = ScorerDictConfig(scorers=_score_kwargs(cfg))
    original_hash = hash(scorer)
    scorer(
        y_true=[1, 0, 1, 1],
        y_pred=[1, 0, 0, 1],
        y_proba=[0.9, 0.1, 0.3, 0.8],
        mode=None,
    )
    scorer.score_dict["extra"] = 42
    assert hash(scorer) == original_hash


def test_scorer_dict_config_equal_content_produces_equal_hash():
    cfg = _compose("default", overrides=["score=classification"])
    raw = _score_kwargs(cfg)
    scorer_a = ScorerDictConfig(scorers=raw)
    scorer_b = ScorerDictConfig(scorers=raw)
    assert hash(scorer_a) == hash(scorer_b)


def test_scorer_dict_config_scores_persist_and_reload():
    cfg = _compose("default", overrides=["score=classification"])
    scorer = ScorerDictConfig(scorers=_score_kwargs(cfg))
    scores = scorer(
        y_true=[1, 0, 1, 1],
        y_pred=[1, 0, 0, 1],
        y_proba=[0.9, 0.1, 0.3, 0.8],
        mode=None,
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "scorer_scores.json"
        scorer.save_scores(scores, path)
        loaded = scorer.load_scores(str(path))
    assert "accuracy" in loaded


def test_scorer_dict_config_object_pickle_roundtrip():
    cfg = _compose("default", overrides=["score=classification"])
    scorer = ScorerDictConfig(scorers=_score_kwargs(cfg))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "scorer.pkl"
        scorer.save_object(scorer, str(path))
        loaded = scorer.load_object(str(path))
    assert isinstance(loaded, ScorerDictConfig)


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
def test_score_compose_via_optimize_cfg_job_cli():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deckard",
            "optimize",
            "--cfg",
            "job",
            *SELECTED_SCORE_OVERRIDES,
        ],
        cwd=str(EXAMPLES_SKLEARN_DIR),
        env={
            **make_runtime_env(DECKARD_RC_PATH),
            "DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "score:" in result.stdout
    assert "accuracy" in result.stdout


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/sklearn/.deckard_rc not found",
)
def test_score_experiment_execution_via_optimize_cli(tmp_path):
    study_name = f"score_exec_{uuid4().hex[:8]}"
    storage = f"sqlite:///{(tmp_path / 'score_exec.db').as_posix()}"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deckard",
            "optimize",
            "--multirun",
            *SELECTED_SCORE_OVERRIDES,
            "hydra.sweeper.n_trials=1",
            "hydra.sweeper.n_jobs=1",
            f"hydra.sweeper.study_name={study_name}",
            f"hydra.sweeper.storage={storage}",
        ],
        cwd=str(EXAMPLES_SKLEARN_DIR),
        env={
            **make_runtime_env(DECKARD_RC_PATH),
            "DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION": "1",
        },
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    study = optuna.load_study(study_name=study_name, storage=storage)
    assert len(study.get_trials(deepcopy=False)) >= 1
