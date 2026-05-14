from deckard.data.fairlearn import FairlearnDataConfig
from deckard.data.lifelines import LifelinesDataConfig
from deckard.data import PytorchDataPipelineConfig
from deckard.data.pipeline import DefaultDataPipelineConfig
from deckard.plugins.anjana.data import AnjanaDataPipelineConfig
from deckard.plugins.fairlearn.data import FairlearnDataPipelineConfig
from deckard.pytorch.data import PytorchDataPipelineConfig as PytorchDataPipelineConfigFromUserFacing


def test_data_family_aliases_are_importable():
    assert FairlearnDataConfig is not None
    assert LifelinesDataConfig is not None


def test_data_pipeline_family_aliases_are_importable():
    assert DefaultDataPipelineConfig is not None
    assert AnjanaDataPipelineConfig is not None
    assert FairlearnDataPipelineConfig is not None
    assert PytorchDataPipelineConfig is not None


def test_data_pipeline_wrapper_is_reexported_at_top_level():
    assert PytorchDataPipelineConfig is PytorchDataPipelineConfigFromUserFacing


def test_data_pipeline_default_configs_are_constructible():
    assert isinstance(DefaultDataPipelineConfig().pipeline, dict)
    assert isinstance(AnjanaDataPipelineConfig().pipeline, dict)
    assert isinstance(FairlearnDataPipelineConfig().pipeline, dict)
