from deckard.data import DataConfig, PytorchDataConfig
from deckard.frameworks.pytorch.data import (
    PytorchDataConfig as PytorchDataConfigFromFramework,
)
from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.fairlearn.data import FairlearnDataConfig
from deckard.plugins.lifelines.data import LifelinesDataConfig


def test_data_family_aliases_are_importable():
    assert FairlearnDataConfig is not None
    assert LifelinesDataConfig is not None


def test_data_pipeline_family_aliases_are_importable():
    assert DataConfig is not None
    assert AnjanaDataConfig is not None
    assert FairlearnDataConfig is not None
    assert PytorchDataConfig is not None


def test_data_pipeline_pytorch_config_matches_framework_canonical_export():
    assert PytorchDataConfig is PytorchDataConfigFromFramework
