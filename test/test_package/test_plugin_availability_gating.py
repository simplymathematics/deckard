import importlib

import pytest

import deckard.data as data_package
import deckard.experiment as experiment_package
import deckard.model as model_package
import deckard.score as score_package
import deckard.declarations as declarations_mod
import deckard.plugins as plugin_namespace


def test_get_plugin_rejects_missing_optional_dependencies(monkeypatch):
    original = plugin_namespace.is_plugin_available

    def _fake_is_plugin_available(name: str) -> bool:
        if name == "fairlearn":
            return False
        return original(name)

    monkeypatch.setattr(
        plugin_namespace,
        "is_plugin_available",
        _fake_is_plugin_available,
    )

    with pytest.raises(ImportError, match="fairlearn"):
        plugin_namespace.get_plugin("fairlearn")


def test_data_reexports_follow_plugin_availability(monkeypatch):
    monkeypatch.setattr(
        declarations_mod,
        "is_package_available",
        lambda name: False if name in {"fairlearn", "anjana", "pycanon"} else True,
    )

    reloaded = importlib.reload(data_package)
    try:
        assert "FairlearnDataConfig" not in reloaded.__all__
        assert "AnjanaDataConfig" not in reloaded.__all__
        assert not hasattr(reloaded, "FairlearnDataConfig")
        assert not hasattr(reloaded, "AnjanaDataConfig")
    finally:
        importlib.reload(data_package)


def test_top_level_plugin_aliases_match_data_modules():
    if not plugin_namespace.is_plugin_available("anjana"):
        pytest.skip("anjana optional dependencies are not installed")
    if not plugin_namespace.is_plugin_available("fairlearn"):
        pytest.skip("fairlearn optional dependencies are not installed")

    import deckard.plugins.anjana as anjana_package
    import deckard.plugins.fairlearn as fairlearn_package

    from deckard.plugins.anjana.data import AnjanaDataConfig, PrivacyBehaviorMixin
    from deckard.plugins.fairlearn.data import (
        FairlearnDataConfig,
        FairnessBehaviorMixin,
    )

    assert anjana_package.AnjanaDataConfig is AnjanaDataConfig
    assert anjana_package.PrivacyBehaviorMixin is PrivacyBehaviorMixin
    assert fairlearn_package.FairlearnDataConfig is FairlearnDataConfig
    assert fairlearn_package.FairnessBehaviorMixin is FairnessBehaviorMixin


def test_top_level_lifelines_alias_matches_module_export():
    if not plugin_namespace.is_plugin_available("lifelines"):
        pytest.skip("lifelines optional dependencies are not installed")

    import deckard.plugins.lifelines as lifelines_package

    from deckard.plugins.lifelines.experiment import SurvivalExperimentConfig

    assert lifelines_package.SurvivalExperimentConfig is SurvivalExperimentConfig


@pytest.mark.parametrize(
    ("package", "missing_exports"),
    [
        (
            model_package,
            (
                "FairlearnModelConfig",
                "AnjanaModelConfig",
                "PytorchModelConfig",
                "SurvivalModelConfig",
            ),
        ),
        (
            experiment_package,
            (
                "TorchExperimentConfig",
                "SurvivalExperimentConfig",
            ),
        ),
    ],
)
def test_core_reexports_follow_optional_availability(
    monkeypatch,
    package,
    missing_exports: tuple[str, ...],
):
    monkeypatch.setattr(plugin_namespace, "is_plugin_available", lambda name: False)
    monkeypatch.setattr(declarations_mod, "is_package_available", lambda name: False)

    reloaded = importlib.reload(package)
    try:
        for export_name in missing_exports:
            assert export_name not in reloaded.__all__
    finally:
        importlib.reload(package)


def test_score_exports_follow_plugin_availability(monkeypatch):
    monkeypatch.setattr(
        declarations_mod,
        "is_package_available",
        lambda name: (
            False if name in {"fairlearn", "anjana", "pycanon", "lifelines"} else True
        ),
    )

    reloaded = importlib.reload(score_package)
    try:
        assert "DefaultFairlearnScorerDictConfig" not in reloaded.__all__
        assert "DefaultAnjanaScorerDictConfig" not in reloaded.__all__
        assert "DefaultLifelinesConfig" not in reloaded.__all__
    finally:
        importlib.reload(score_package)
