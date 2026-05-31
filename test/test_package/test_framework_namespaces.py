from deckard.frameworks.pytorch import (
    DefenseConfig as PytorchDefenseConfig,
    PytorchAttackConfig,
    PytorchDataConfig as FrameworkPytorchDataConfig,
    PytorchModelConfig,
    TorchExperimentConfig,
)
from deckard.frameworks.transformers import GenericFlexibleTransformer
from deckard.frameworks.sklearn import (
    DefenseConfig as SklearnDefenseConfig,
)
from deckard.pytorch.attack import PytorchAttackConfig as NamespacePytorchAttackConfig
from deckard.pytorch.data import (
    PytorchCustomDataConfig as NamespacePytorchCustomDataConfig,
    PytorchDataConfig as NamespacePytorchDataConfig,
)
from deckard.pytorch.data_pipeline import (
    PytorchDataConfig as PipelinePytorchDataConfig,
)
from deckard.pytorch.experiment import (
    TorchExperimentConfig as NamespaceTorchExperimentConfig,
)
from deckard.pytorch.fairness_data import (
    FairlearnPytorchDataConfig as NamespaceFairlearnPytorchDataConfig,
)
from deckard.pytorch.model import PytorchModelConfig as NamespacePytorchModelConfig


def test_framework_namespace_aliases_are_importable():
    assert PytorchModelConfig is not None
    assert SklearnDefenseConfig is not None
    assert PytorchDefenseConfig is not None
    assert GenericFlexibleTransformer is not None


def test_pytorch_namespace_wrapper_aliases_match_framework_symbols():
    assert NamespacePytorchAttackConfig is PytorchAttackConfig
    assert NamespacePytorchDataConfig is FrameworkPytorchDataConfig
    assert PipelinePytorchDataConfig is FrameworkPytorchDataConfig
    assert NamespaceTorchExperimentConfig is TorchExperimentConfig
    assert NamespacePytorchModelConfig is PytorchModelConfig
    assert NamespacePytorchCustomDataConfig is not None
    assert NamespaceFairlearnPytorchDataConfig is not None
