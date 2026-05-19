from deckard.plugins.fairlearn.model import FairlearnModelConfig
from deckard.plugins.lifelines.model import SurvivalModelConfig
from deckard.model import (
    DefaultDefenseConfig,
    DefaultPytorchDefenseConfig,
    DefaultSklearnDefenseConfig,
)
from deckard.model.defense.default import (
    DefaultDefenseConfig as DefaultDefenseConfigFromModule,
)
from deckard.frameworks.pytorch.defense import (
    DefaultPytorchDefenseConfig as DefaultPytorchDefenseConfigFromModule,
)
from deckard.frameworks.sklearn.defense import (
    DefaultSklearnDefenseConfig as DefaultSklearnDefenseConfigFromModule,
)


def test_model_family_aliases_are_importable():
    assert FairlearnModelConfig is not None
    assert SurvivalModelConfig is not None


def test_model_defense_package_exports_are_importable():
    assert DefaultDefenseConfig is DefaultDefenseConfigFromModule
    assert DefaultSklearnDefenseConfig is DefaultSklearnDefenseConfigFromModule
    assert DefaultPytorchDefenseConfig is DefaultPytorchDefenseConfigFromModule


def test_default_defense_config_is_neutral_baseline():
    cfg = DefaultDefenseConfig()
    assert cfg.defense_name is None
    assert cfg.init_params["class"] == "baseline"


def test_default_sklearn_defense_config_sets_framework_marker():
    cfg = DefaultSklearnDefenseConfig()
    assert cfg.defense_name is None
    assert cfg.init_params["class"] == "sklearn.baseline"


def test_default_pytorch_defense_config_sets_framework_marker():
    cfg = DefaultPytorchDefenseConfig()
    assert cfg.defense_name is None
    assert cfg.init_params["class"] == "pytorch.baseline"
