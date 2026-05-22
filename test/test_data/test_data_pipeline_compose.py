from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from deckard.data.pipeline.base import (
    DataConfig,
    DataConfig,
    DataConfig,
)


def _compose_from_dir(config_dir: Path, config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name)


def test_sklearn_pipeline_yaml_configs_compose_and_instantiate():
    config_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "sklearn"
        / "config"
        / "data"
        / "pipeline"
    )

    default_cfg = _compose_from_dir(config_dir, "default_pipeline")
    anjana_cfg = _compose_from_dir(config_dir, "anjana_pipeline")
    fairlearn_cfg = _compose_from_dir(config_dir, "fairlearn_pipeline")

    assert isinstance(instantiate(default_cfg), DataConfig)
    assert isinstance(instantiate(anjana_cfg), DataConfig)
    assert isinstance(instantiate(fairlearn_cfg), DataConfig)


def test_pytorch_pipeline_yaml_config_compose_and_instantiate():
    import pytest

    pytest.importorskip("torch")
    from deckard.frameworks.pytorch.data import PytorchDataConfig

    config_dir = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "pytorch"
        / "config"
        / "data"
        / "pipeline"
    )

    pytorch_cfg = _compose_from_dir(config_dir, "pytorch_pipeline")
    assert isinstance(instantiate(pytorch_cfg), PytorchDataConfig)
