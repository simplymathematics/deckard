import importlib
import sys
from types import ModuleType

import pytest


def test_fairness_pytorch_compat_reexport_matches_framework_symbol():
    compat_mod = importlib.import_module("deckard.data.fairness_pytorch")
    framework_mod = importlib.import_module("deckard.frameworks.pytorch.fairness_data")

    assert compat_mod.TinyFairness is framework_mod.TinyFairness
    assert compat_mod.__all__ == ["TinyFairness"]


def test_data_pipeline_core_compat_reexports_base_symbols():
    compat_mod = importlib.import_module("deckard.data.pipeline.core")
    base_mod = importlib.import_module("deckard.data.pipeline.base")

    assert compat_mod.DataPipeline is base_mod.DataPipeline
    assert compat_mod.DataConfig is base_mod.DataConfig
    assert set(compat_mod.__all__) == {"DataPipeline", "DataConfig"}


def test_plot_declarations_expose_expected_plugins_and_defaults():
    declarations = importlib.import_module("deckard.plot.declarations")

    assert declarations.PLOT_DEFAULT["backend"] == "yellowbrick"
    assert "roc_auc" in declarations.PLOT_TYPES
    assert declarations.SEABORN_PLOTTER_PLUGIN.backend == "seaborn"
    assert declarations.YELLOWBRICK_PLOTTER_PLUGIN.backend == "yellowbrick"


def test_yellowbrick_compat_module_reexports_symbols(monkeypatch):
    fake_plot_mod = ModuleType("deckard.plugins.yellowbrick.plot")
    fake_plot_mod.YellowbrickPlotConfig = object()
    fake_plot_mod.YellowbrickConfigList = object()

    monkeypatch.setitem(sys.modules, "deckard.plugins.yellowbrick.plot", fake_plot_mod)

    compat_mod = importlib.import_module("deckard.plot.yellowbrick_plots")
    compat_mod = importlib.reload(compat_mod)

    assert compat_mod.YellowbrickPlotConfig is fake_plot_mod.YellowbrickPlotConfig
    assert compat_mod.YellowbrickConfigList is fake_plot_mod.YellowbrickConfigList
    assert compat_mod.__all__ == ["YellowbrickPlotConfig", "YellowbrickConfigList"]


def test_transformers_namespace_getattr_dispatches_known_symbols(monkeypatch):
    pkg = importlib.import_module("deckard.frameworks.transformers")

    fake_decl = ModuleType("deckard.frameworks.transformers.declarations")
    sentinel_transformer = object()
    sentinel_wrapper = object()
    fake_decl.GenericFlexibleTransformer = sentinel_transformer
    fake_decl.HuggingFaceArtModelWrapper = sentinel_wrapper

    fake_model = ModuleType("deckard.frameworks.transformers.model")
    sentinel_model_cfg = object()
    fake_model.HuggingFacePytorchModelConfig = sentinel_model_cfg

    monkeypatch.setitem(sys.modules, "deckard.frameworks.transformers.declarations", fake_decl)
    monkeypatch.setitem(sys.modules, "deckard.frameworks.transformers.model", fake_model)

    assert pkg.__getattr__("GenericFlexibleTransformer") is sentinel_transformer
    assert pkg.__getattr__("HuggingFaceArtModelWrapper") is sentinel_wrapper
    assert pkg.__getattr__("HuggingFacePytorchModelConfig") is sentinel_model_cfg


def test_transformers_namespace_getattr_rejects_unknown_symbol():
    pkg = importlib.import_module("deckard.frameworks.transformers")

    with pytest.raises(AttributeError, match="has no attribute"):
        pkg.__getattr__("definitely_missing_symbol")


def test_transformers_model_module_is_importable():
    model_mod = importlib.import_module("deckard.frameworks.transformers.model")

    assert hasattr(model_mod, "HuggingFacePytorchModelConfig")