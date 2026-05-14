from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.fairlearn.model import FairlearnModelConfig
from deckard.plugins.lifelines.experiment import SurvivalExperimentConfig
from deckard.plugins.yellowbrick.plot import YellowbrickPlotConfig


def test_plugin_family_modules_export_public_configs():
    assert AnjanaDataConfig is not None
    assert FairlearnModelConfig is not None
    assert SurvivalExperimentConfig is not None
    assert YellowbrickPlotConfig is not None
