from __future__ import annotations

import sys
from types import ModuleType

import pytest

import deckard.frameworks.pytorch as pytorch_pkg


def _stub_module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.mark.parametrize(
    ("symbol", "module_name", "attr_name"),
    [
        ("PytorchDataConfig", "deckard.frameworks.pytorch.data", "PytorchDataConfig"),
        (
            "PytorchCustomDataConfig",
            "deckard.frameworks.pytorch.data",
            "PytorchCustomDataConfig",
        ),
        (
            "FairlearnPytorchDataConfig",
            "deckard.frameworks.pytorch.fairness_data",
            "FairlearnPytorchDataConfig",
        ),
        (
            "PytorchModelConfig",
            "deckard.frameworks.pytorch.model",
            "PytorchModelConfig",
        ),
        (
            "PytorchAttackConfig",
            "deckard.frameworks.pytorch.attack",
            "PytorchAttackConfig",
        ),
        (
            "TorchExperimentConfig",
            "deckard.frameworks.pytorch.experiment",
            "TorchExperimentConfig",
        ),
        (
            "PytorchBaseSampler",
            "deckard.frameworks.pytorch.sample",
            "PytorchBaseSampler",
        ),
        (
            "PytorchSplitSampler",
            "deckard.frameworks.pytorch.sample",
            "PytorchSplitSampler",
        ),
        (
            "PytorchFoldSampler",
            "deckard.frameworks.pytorch.sample",
            "PytorchFoldSampler",
        ),
        (
            "PytorchShuffleSampler",
            "deckard.frameworks.pytorch.sample",
            "PytorchShuffleSampler",
        ),
        (
            "TorchBaseSampler",
            "deckard.frameworks.pytorch.sample",
            "TorchBaseSampler",
        ),
        (
            "TorchSplitSampler",
            "deckard.frameworks.pytorch.sample",
            "TorchSplitSampler",
        ),
        (
            "TorchKFoldSampler",
            "deckard.frameworks.pytorch.sample",
            "TorchKFoldSampler",
        ),
        (
            "TorchShuffleSampler",
            "deckard.frameworks.pytorch.sample",
            "TorchShuffleSampler",
        ),
    ],
)
def test_pytorch_package_getattr_resolves_lazy_symbols(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
    module_name: str,
    attr_name: str,
) -> None:
    sentinel = type(symbol, (), {})
    module = _stub_module(module_name, **{attr_name: sentinel})
    if module_name.endswith(".sample"):
        for alias in (
            "PytorchBaseSampler",
            "PytorchSplitSampler",
            "PytorchFoldSampler",
            "PytorchShuffleSampler",
            "TorchBaseSampler",
            "TorchSplitSampler",
            "TorchKFoldSampler",
            "TorchShuffleSampler",
        ):
            setattr(
                module,
                alias,
                sentinel if alias == attr_name else type(alias, (), {}),
            )
    if module_name.endswith(".data"):
        for alias in ("PytorchDataConfig", "PytorchCustomDataConfig"):
            setattr(
                module,
                alias,
                sentinel if alias == attr_name else type(alias, (), {}),
            )

    monkeypatch.setitem(sys.modules, module_name, module)

    assert pytorch_pkg.__getattr__(symbol) is sentinel


def test_pytorch_package_getattr_returns_defense_config() -> None:
    assert pytorch_pkg.__getattr__("DefenseConfig") is pytorch_pkg.DefenseConfig


def test_pytorch_package_getattr_rejects_unknown_symbol() -> None:
    with pytest.raises(AttributeError):
        pytorch_pkg.__getattr__("DefinitelyMissingPytorchExport")
