from deckard.attack import AttackConfig
from deckard.data import DataConfig
from deckard.detector import DetectorConfig
from deckard.experiment import ExperimentConfig
from deckard.model import DefenseConfig, ModelConfig
from deckard.score import ScorerDictConfig


def test_core_config_public_accessors_exist() -> None:
    assert isinstance(DataConfig.X, property)
    assert isinstance(DataConfig.y, property)
    assert isinstance(DataConfig.split_indices, property)
    assert isinstance(DataConfig.sensitive_train, property)
    assert isinstance(DataConfig.sensitive_test, property)
    assert isinstance(DataConfig.sensitive_val, property)
    assert isinstance(DataConfig.sensitive_all, property)

    assert isinstance(ModelConfig.model, property)
    assert isinstance(ModelConfig.fitted_estimator, property)
    assert isinstance(ModelConfig.test_predictions, property)
    assert isinstance(ModelConfig.test_probabilities, property)
    assert isinstance(ModelConfig.defense_pipeline, property)

    assert isinstance(DefenseConfig.model, property)
    assert isinstance(DefenseConfig.model_config, property)

    assert isinstance(AttackConfig.attack_family, property)
    assert isinstance(AttackConfig.attack_subtype, property)
    assert isinstance(AttackConfig.attack_kind, property)
    assert isinstance(AttackConfig.attack_instance, property)

    assert isinstance(DetectorConfig.detector, property)
    assert isinstance(DetectorConfig.detector_instance, property)
    assert isinstance(ExperimentConfig.attack_chain, property)
    assert isinstance(ExperimentConfig.runtime_scores, property)

    assert isinstance(ScorerDictConfig.configured_scorers, property)
