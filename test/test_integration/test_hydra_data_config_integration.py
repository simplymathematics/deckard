from omegaconf import OmegaConf

from .shared_compose import compose_pytorch, compose_sklearn


def test_sklearn_data_profile_anjana_composes():
    cfg = compose_sklearn("data/anjana")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    assert data_cfg["name"] == "make_classification"
    assert data_cfg["_target_"] == "deckard.plugins.anjana.AnjanaDataConfig"
    assert data_cfg["alias"] == "anjana"


def test_sklearn_data_profile_fairlearn_composes():
    cfg = compose_sklearn("data/fair-adult")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    assert data_cfg["name"].endswith("raw_data/adult_income/adult_income_dataset.csv")
    assert data_cfg["_target_"] == "deckard.plugins.fairlearn.FairlearnDataConfig"
    assert data_cfg["sensitive_columns"] == ["sex"]


def test_sklearn_data_profile_lifelines_composes():
    cfg = compose_sklearn("data/lung")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    assert data_cfg["name"] == "lung"
    assert data_cfg["target"] == "E"
    assert data_cfg["classifier"] is False


def test_pytorch_data_profile_torch_mnist_composes():
    cfg = compose_pytorch("data/torch_mnist")
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)

    assert data_cfg["name"] == "torchvision.datasets.MNIST"
    assert data_cfg["_target_"] == "deckard.frameworks.pytorch.data.PytorchDataConfig"
    assert data_cfg["alias"] == "torch_mnist"
