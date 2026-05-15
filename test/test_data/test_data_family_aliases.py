from deckard.data.fairlearn import FairlearnDataConfig
from deckard.data.lifelines import LifelinesDataConfig
from deckard.data import PytorchDataConfig
from deckard.data import DataConfig
from deckard.data.pipeline import DataPipelineConfig
from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.fairlearn.data import FairlearnDataConfig
from deckard.pytorch.data import PytorchDataConfig as PytorchDataConfigFromUserFacing


def test_data_family_aliases_are_importable():
    assert FairlearnDataConfig is not None
    assert LifelinesDataConfig is not None


def test_data_pipeline_family_aliases_are_importable():
    assert DataConfig is not None
    assert AnjanaDataConfig is not None
    assert FairlearnDataConfig is not None
    assert PytorchDataConfig is not None


def test_data_pipeline_wrapper_is_reexported_at_top_level():
    assert PytorchDataConfig is PytorchDataConfigFromUserFacing
