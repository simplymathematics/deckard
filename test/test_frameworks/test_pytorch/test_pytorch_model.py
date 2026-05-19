import pytest

torch = pytest.importorskip("torch")
Tensor = pytest.importorskip("torch").Tensor
PytorchDataConfig = pytest.importorskip(
    "deckard.frameworks.pytorch.data",
).PytorchDataConfig
PytorchCustomDataConfig = pytest.importorskip(
    "deckard.frameworks.pytorch.data",
).PytorchCustomDataConfig
