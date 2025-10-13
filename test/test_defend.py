import pytest
import argparse
from deckard.defend import defense_main
from deckard.data import DataConfig
from deckard.model import ModelConfig
from deckard.attack import AttackConfig
from deckard.defend import DefenseConfig


class DummyArgs(argparse.Namespace):
    def __init__(self):
        super().__init__()
        self.defense_config_file = None
        self.defense_config_params = None
        self.defense_score_filepath = None
        self.data_config_file = None
        self.data_config_params = None
        self.model_config_file = None
        self.model_config_params = None
        self.attack_config_file = None
        self.attack_config_params = None


@pytest.fixture
def dummy_args(tmp_path):
    args = DummyArgs()
    args.defense_score_filepath = str(tmp_path / "scores.pkl")
    return args


@pytest.fixture
def patch_configs(monkeypatch):
    # Patch initialize_data_config, initialize_model_config, initialize_defense_config, initialize_attack_config
    monkeypatch.setattr(
        "deckard.defend.initialize_data_config",
        lambda args: DataConfig(),
    )
    monkeypatch.setattr(
        "deckard.defend.initialize_model_config",
        lambda args: ModelConfig(),
    )
    monkeypatch.setattr(
        "deckard.defend.initialize_defense_config",
        lambda args: DefenseConfig(),
    )
    monkeypatch.setattr(
        "deckard.defend.initialize_attack_config",
        lambda args: AttackConfig(),
    )
    # Patch call parsers to return dummy args
    monkeypatch.setattr("deckard.defend.data_call_parser", argparse.ArgumentParser())
    monkeypatch.setattr("deckard.defend.model_call_parser", argparse.ArgumentParser())
    monkeypatch.setattr("deckard.defend.attack_call_parser", argparse.ArgumentParser())
    monkeypatch.setattr("deckard.defend.defense_call_parser", argparse.ArgumentParser())
    # Patch score_dict for configs
    DataConfig.score_dict = {"data_score": 1}
    ModelConfig.score_dict = {"model_score": 2}
    DefenseConfig.score_dict = {"defense_score": 3}
    AttackConfig.score_dict = {"attack_score": 4}


def test_defense_main_scores(dummy_args, patch_configs, tmp_path):
    data = DataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 100,
            "n_features": 20,
            "n_informative": 15,
            "random_state": 42,
            "n_classes": 3,
        },
    )
    data()
    defense_config = DefenseConfig(
        model_type="sklearn.ensemble.RandomForestClassifier",
        model_params={"n_estimators": 10, "random_state": 42},
    )
    score_dict = defense_config(data=data)
    assert isinstance(score_dict, dict)
    assert "accuracy" in score_dict or "mse" in score_dict


def test_defense_main_no_score_file(monkeypatch):
    args = DummyArgs()
    # Patch configs and parsers
    monkeypatch.setattr(
        "deckard.defend.initialize_data_config",
        lambda args: DataConfig(),
    )
    monkeypatch.setattr(
        "deckard.defend.initialize_model_config",
        lambda args: ModelConfig(),
    )
    monkeypatch.setattr(
        "deckard.defend.initialize_defense_config",
        lambda args: DefenseConfig(),
    )
    monkeypatch.setattr(
        "deckard.defend.initialize_attack_config",
        lambda args: AttackConfig(),
    )
    monkeypatch.setattr("deckard.defend.data_call_parser", argparse.ArgumentParser())
    monkeypatch.setattr("deckard.defend.model_call_parser", argparse.ArgumentParser())
    monkeypatch.setattr("deckard.defend.attack_call_parser", argparse.ArgumentParser())
    monkeypatch.setattr("deckard.defend.defense_call_parser", argparse.ArgumentParser())
    DataConfig.score_dict = {"data_score": 10}
    ModelConfig.score_dict = {"model_score": 20}
    DefenseConfig.score_dict = {"defense_score": 30}
    AttackConfig.score_dict = {"attack_score": 40}
    scores = defense_main(args)
    assert isinstance(scores, dict)
    assert "accuracy" in scores
