from deckard.frameworks.pytorch import (
    DefaultPytorchDefenseConfig,
    PytorchModelConfig,
)
from deckard.frameworks.sklearn import (
    DefaultSklearnDefenseConfig,
)


def test_framework_namespace_aliases_are_importable():
    assert PytorchModelConfig is not None
    assert DefaultSklearnDefenseConfig is not None
    assert DefaultPytorchDefenseConfig is not None
