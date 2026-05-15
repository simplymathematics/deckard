from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.yellowbrick import YellowbrickPlotConfig


def test_plugin_family_exports_are_importable_without_full_runtime_execution():
    assert AnjanaDataConfig is not None
    assert YellowbrickPlotConfig is not None
