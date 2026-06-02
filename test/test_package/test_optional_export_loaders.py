from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

import deckard.experiment as experiment_pkg
import deckard.model as model_pkg
import deckard._optional as optional_registry


def _install_fake_module(
    module_name: str,
    attrs: dict[str, object],
) -> ModuleType | None:
    previous = sys.modules.get(module_name)
    module = ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return previous


def _restore_module(module_name: str, previous: ModuleType | None) -> None:
    if previous is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous


@pytest.mark.parametrize(
    ("package", "loader_names"),
    [
        (
            model_pkg,
            (
                "_load_fairlearn_model_symbols",
                "_load_lifelines_model_symbols",
                "_load_anjana_model_symbols",
                "_load_torch_model_symbols",
            ),
        ),
        (
            experiment_pkg,
            (
                "_load_torch_experiment_symbols",
                "_load_lifelines_experiment_symbols",
            ),
        ),
    ],
)
def test_optional_loaders_return_false_when_unavailable(
    monkeypatch,
    package,
    loader_names: tuple[str, ...],
):
    monkeypatch.setattr(
        optional_registry,
        "is_optional_family_available",
        lambda name, kind=None: False,
    )

    for loader_name in loader_names:
        assert getattr(package, loader_name)() is False


def test_model_optional_loaders_populate_exports(monkeypatch):
    monkeypatch.setattr(
        optional_registry,
        "is_optional_family_available",
        lambda name, kind=None: True,
    )

    class _FairlearnDefenseConfig:
        pass

    class _FairlearnModelConfig:
        pass

    class _FairlearnPytorchModelConfig:
        pass

    class _SurvivalModelConfig:
        pass

    class _AnjanaModelConfig:
        pass

    class _PytorchModelConfig:
        pass

    originals: dict[str, ModuleType | None] = {}

    originals["deckard.plugins.fairlearn.model"] = _install_fake_module(
        "deckard.plugins.fairlearn.model",
        {
            "FairlearnDefenseConfig": _FairlearnDefenseConfig,
            "FairlearnModelConfig": _FairlearnModelConfig,
            "FairlearnPytorchModelConfig": _FairlearnPytorchModelConfig,
        },
    )
    originals["deckard.plugins.lifelines.model"] = _install_fake_module(
        "deckard.plugins.lifelines.model",
        {"SurvivalModelConfig": _SurvivalModelConfig},
    )
    originals["deckard.plugins.anjana.model"] = _install_fake_module(
        "deckard.plugins.anjana.model",
        {"AnjanaModelConfig": _AnjanaModelConfig},
    )
    originals["deckard.frameworks.pytorch.model"] = _install_fake_module(
        "deckard.frameworks.pytorch.model",
        {"PytorchModelConfig": _PytorchModelConfig},
    )

    reloaded = importlib.reload(model_pkg)
    try:
        monkeypatch.setattr(
            optional_registry,
            "is_optional_family_available",
            lambda name, kind=None: True,
        )

        assert reloaded._load_fairlearn_model_symbols() is True
        assert reloaded._load_lifelines_model_symbols() is True
        assert reloaded._load_anjana_model_symbols() is True
        assert reloaded._load_torch_model_symbols() is True

        assert reloaded.__getattr__("FairlearnModelConfig") is _FairlearnModelConfig
        assert reloaded.__getattr__("SurvivalModelConfig") is _SurvivalModelConfig
        assert reloaded.__getattr__("AnjanaModelConfig") is _AnjanaModelConfig
        assert reloaded.__getattr__("PytorchModelConfig") is _PytorchModelConfig
    finally:
        importlib.reload(model_pkg)
        for module_name, previous in originals.items():
            _restore_module(module_name, previous)


def test_experiment_optional_loaders_populate_exports(monkeypatch):
    monkeypatch.setattr(
        optional_registry,
        "is_optional_family_available",
        lambda name, kind=None: True,
    )

    class _TorchExperimentConfig:
        pass

    class _SurvivalExperimentConfig:
        pass

    originals: dict[str, ModuleType | None] = {}

    originals["deckard.frameworks.pytorch.experiment"] = _install_fake_module(
        "deckard.frameworks.pytorch.experiment",
        {"TorchExperimentConfig": _TorchExperimentConfig},
    )
    originals["deckard.plugins.lifelines.experiment"] = _install_fake_module(
        "deckard.plugins.lifelines.experiment",
        {"SurvivalExperimentConfig": _SurvivalExperimentConfig},
    )

    reloaded = importlib.reload(experiment_pkg)
    try:
        monkeypatch.setattr(
            optional_registry,
            "is_optional_family_available",
            lambda name, kind=None: True,
        )

        assert reloaded._load_torch_experiment_symbols() is True
        assert reloaded._load_lifelines_experiment_symbols() is True

        assert reloaded.__getattr__("TorchExperimentConfig") is _TorchExperimentConfig
        assert (
            reloaded.__getattr__("SurvivalExperimentConfig")
            is _SurvivalExperimentConfig
        )

        with pytest.raises(AttributeError):
            reloaded.__getattr__("DefinitelyMissingExperimentExport")
    finally:
        importlib.reload(experiment_pkg)
        for module_name, previous in originals.items():
            _restore_module(module_name, previous)
