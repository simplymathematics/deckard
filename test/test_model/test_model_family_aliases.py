from deckard.model import (
    DefenseConfig,
)
from deckard.model.defense.default import (
    DefenseConfig as DefenseConfigFromModule,
)
from deckard.frameworks.pytorch.defense import DefenseConfig as PytorchDefenseConfig
from deckard.frameworks.sklearn.defense import DefenseConfig as SklearnDefenseConfig
from deckard.plugins.fairlearn.model import FairlearnModelConfig
from deckard.plugins.lifelines.model import SurvivalModelConfig


def test_model_family_aliases_are_importable():
    assert FairlearnModelConfig is not None
    assert SurvivalModelConfig is not None


def test_model_defense_package_exports_are_importable():
    assert DefenseConfig is DefenseConfigFromModule
    assert SklearnDefenseConfig is DefenseConfig
    assert PytorchDefenseConfig is DefenseConfig


def test_defense_config_is_neutral_baseline():
    cfg = DefenseConfig()
    assert cfg.name is None
