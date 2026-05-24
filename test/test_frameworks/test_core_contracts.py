"""Focused tests confirming runtime configs are decoupled from framework contracts."""

import pytest

from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.detector import DetectorConfig
from deckard.experiment import ExperimentConfig
from deckard.model import ModelConfig
from deckard.score import ScorerDictConfig
from deckard.utils import BaseConfig


@pytest.mark.parametrize(
    "runtime_cls",
    [
        DataConfig,
        ModelConfig,
        AttackConfig,
        DetectorConfig,
        ExperimentConfig,
        ScorerDictConfig,
    ],
)
def test_runtime_configs_inherit_config_base(runtime_cls):
    assert issubclass(
        runtime_cls,
        BaseConfig,
    ), f"{runtime_cls.__name__} must inherit ConfigBase"
