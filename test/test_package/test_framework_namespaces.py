from deckard.frameworks.pytorch import (
    DefenseConfig as PytorchDefenseConfig,
    PytorchModelConfig,
)
from deckard.frameworks.sklearn import (
    DefenseConfig as SklearnDefenseConfig,
)


def test_framework_namespace_aliases_are_importable():
    assert PytorchModelConfig is not None
    assert SklearnDefenseConfig is not None
    assert PytorchDefenseConfig is not None
