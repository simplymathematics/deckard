from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

SKLEARN_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
PYTORCH_CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "config"


def _compose_sklearn(config_name: str, overrides: list[str] | None = None):
    overrides = overrides or []
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_store = ConfigStore.instance()
    for key in list(config_store.repo.keys()):
        if key not in {"hydra", "_dummy_empty_config_.yaml"}:
            config_store.repo.pop(key, None)
    with initialize_config_dir(version_base="1.3", config_dir=str(SKLEARN_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def _compose_pytorch(config_name: str, overrides: list[str] | None = None):
    overrides = overrides or []
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_store = ConfigStore.instance()
    for key in list(config_store.repo.keys()):
        if key not in {"hydra", "_dummy_empty_config_.yaml"}:
            config_store.repo.pop(key, None)
    with initialize_config_dir(version_base="1.3", config_dir=str(PYTORCH_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def test_sklearn_model_profile_logistic_composes():
    cfg = _compose_sklearn("model/logistic")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["model_type"] == "sklearn.linear_model.LogisticRegression"
    assert model_cfg["classifier"] is True
    assert model_cfg["alias"] == "logistic"


def test_sklearn_model_profile_cox_composes():
    cfg = _compose_sklearn("model/cox")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["model_type"] == "lifelines.fitters.coxph_fitter.CoxPHFitter"
    assert model_cfg["classifier"] is False
    assert model_cfg["alias"] == "cox"


def test_sklearn_default_can_switch_to_test_logistic_model_profile():
    cfg = _compose_sklearn("default", overrides=["model=test-logistic"])
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["model_type"] == "sklearn.linear_model.LogisticRegression"
    assert model_cfg["classifier"] is True
    assert model_cfg["alias"] == "test_logistic"


def test_pytorch_model_profile_tinynet_composes():
    cfg = _compose_pytorch("model/tinynet")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["model_type"] == "deckard.pytorch.model.TinyNet"
    assert model_cfg["classifier"] is True
    assert model_cfg["alias"] == "tinynet"
