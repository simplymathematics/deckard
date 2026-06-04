from omegaconf import OmegaConf

from .shared_compose import compose_pytorch, compose_sklearn


def test_sklearn_model_profile_cox_composes():
    cfg = compose_sklearn("model/cox")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["name"] == "lifelines.fitters.coxph_fitter.CoxPHFitter"
    assert model_cfg["classifier"] is False
    assert model_cfg["alias"] == "cox"


def test_sklearn_default_can_switch_to_test_logistic_model_profile():
    cfg = compose_sklearn("default", overrides=["model=test-logistic"])
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["name"] == "sklearn.linear_model.LogisticRegression"
    assert model_cfg["classifier"] is True
    assert model_cfg["alias"] == "test_logistic"


def test_pytorch_model_profile_tinynet_composes():
    cfg = compose_pytorch("model/tinynet")
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    assert model_cfg["name"] == "deckard.frameworks.pytorch.model.TinyNet"
    assert model_cfg["classifier"] is True
    assert model_cfg["alias"] == "tinynet"
