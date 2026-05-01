from pathlib import Path
import importlib.util

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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


@pytest.mark.skipif(
    importlib.util.find_spec("fairlearn") is None,
    reason="fairlearn is required to validate fairness score profile integration",
)
def test_default_can_switch_to_fairness_score_group():
    cfg = _compose("default", overrides=["data=fair-adult", "score=fairness"])
    score_cfg = OmegaConf.to_container(cfg.score, resolve=True)

    assert "scorers" in score_cfg
    assert "demographic_parity_difference" in score_cfg["scorers"]
    assert "equalized_odds_difference" in score_cfg["scorers"]
