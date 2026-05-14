from deckard.frameworks import (
    FrameworkAttackConfig,
    FrameworkDataConfig,
    FrameworkDetectorConfig,
    FrameworkExperimentConfig,
    FrameworkModelConfig,
    FrameworkScorerConfig,
)


def test_framework_contract_exports_are_importable():
    assert FrameworkDataConfig is not None
    assert FrameworkModelConfig is not None
    assert FrameworkAttackConfig is not None
    assert FrameworkDetectorConfig is not None
    assert FrameworkExperimentConfig is not None
    assert FrameworkScorerConfig is not None
