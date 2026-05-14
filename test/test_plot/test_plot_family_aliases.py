from deckard.plugins.lifelines.plot import SurvivalSeabornPlotterConfig
from deckard.plugins.seaborn.plot import SeabornPlotConfig
from deckard.plugins.yellowbrick.plot import YellowbrickPlotConfig


def test_plot_family_aliases_are_importable():
    assert SeabornPlotConfig is not None
    assert SurvivalSeabornPlotterConfig is not None
    assert YellowbrickPlotConfig is not None
