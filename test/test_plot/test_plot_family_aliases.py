from deckard.Anjana import AnjanaDataConfig as LegacyAnjanaDataConfig
from deckard.plugins.fairlearn.data_pytorch import TinyFairness as LegacyTinyFairness

from deckard.frameworks.pytorch.fairness_data import TinyFairness
from deckard.plot.yellowbrick_plots import (
    YellowbrickPlotConfig as LegacyYellowbrickPlotConfig,
)
from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.lifelines.plot import SurvivalSeabornPlotterConfig
from deckard.plugins.seaborn.plot import SeabornPlotConfig
from deckard.plugins.yellowbrick.plot import YellowbrickPlotConfig


def test_plot_family_aliases_are_importable():
    assert SeabornPlotConfig is not None
    assert SurvivalSeabornPlotterConfig is not None
    assert YellowbrickPlotConfig is not None
    assert LegacyYellowbrickPlotConfig is YellowbrickPlotConfig
    assert LegacyTinyFairness is TinyFairness
    assert LegacyAnjanaDataConfig is AnjanaDataConfig
