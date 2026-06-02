from pathlib import Path

from helpers import reset_hydra_state
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

SKLEARN_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "sklearn" / "config"
)
PYTORCH_CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "pytorch" / "config"
)


def _compose_sklearn(config_name: str, overrides: list[str] | None = None):
    overrides = overrides or []
    reset_hydra_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(SKLEARN_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def _compose_pytorch(config_name: str, overrides: list[str] | None = None):
    overrides = overrides or []
    reset_hydra_state()
    with initialize_config_dir(version_base="1.3", config_dir=str(PYTORCH_CONFIG_DIR)):
        return compose(config_name=config_name, overrides=overrides)


def test_sklearn_data_profile_anjana_composes():
    cfg = _compose_sklearn("data/anjana")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    assert data_cfg["name"] == "make_classification"
    assert data_cfg["_target_"] == "deckard.plugins.anjana.AnjanaDataConfig"
    assert data_cfg["alias"] == "anjana"


def test_sklearn_data_profile_fairlearn_composes():
    cfg = _compose_sklearn("data/fair-adult")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    assert data_cfg["name"] == "sklearn.adult"
    assert data_cfg["_target_"] == "deckard.plugins.fairlearn.FairlearnDataConfig"
    assert data_cfg["sensitive_columns"] == ["sex"]


def test_sklearn_data_profile_lifelines_composes():
    cfg = _compose_sklearn("data/lung")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    assert data_cfg["name"] == "lung"
    assert data_cfg["target"] == "E"
    assert data_cfg["classifier"] is False


def test_pytorch_data_profile_torch_mnist_composes():
    cfg = _compose_pytorch("data/torch_mnist")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    assert data_cfg["name"] == "torchvision.datasets.MNIST"
    assert data_cfg["_target_"] == "deckard.frameworks.pytorch.data.PytorchDataConfig"
    assert data_cfg["alias"] == "torch_mnist"
