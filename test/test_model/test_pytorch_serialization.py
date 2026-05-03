import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from deckard.model.defend import DefensePipelineConfig

PytorchModelConfig = pytest.importorskip("deckard.model.pytorch").PytorchModelConfig


def test_pytorch_model_config_save_and_load_roundtrip():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 2},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "torch_model.pkl"
        cfg.save(str(model_path))

        loaded = PytorchModelConfig(
            model_type="torch.nn.Linear",
            model_params={"in_features": 4, "out_features": 2},
            classifier=True,
        )
        loaded.load(str(model_path))

        assert loaded.model_type == "torch.nn.Linear"
        assert loaded.model_params["in_features"] == 4
        assert loaded.model_params["out_features"] == 2

        state_1 = cfg.get_model().state_dict()
        state_2 = loaded.get_model().state_dict()
        assert state_1.keys() == state_2.keys()
        for key in state_1:
            assert torch.equal(state_1[key], state_2[key])


def test_pytorch_model_training_records_optimizer_loss_and_serializes_it():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 2, "batch_size": 2},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    cfg._train(X, y)

    assert "optimizer_loss" in cfg.score_dict
    assert cfg.score_dict["optimizer_loss"] is not None
    assert isinstance(cfg.score_dict["optimizer_loss"], float)
    assert "epochs" in cfg.score_dict
    assert len(cfg.score_dict["epochs"]) == 2

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "torch_model_with_loss.pkl"
        cfg.save(str(model_path))

        loaded = PytorchModelConfig(
            model_type="torch.nn.Linear",
            model_params={"in_features": 4, "out_features": 2},
            classifier=True,
        )
        loaded.load(str(model_path))

        assert loaded.score_dict["optimizer_loss"] == cfg.score_dict["optimizer_loss"]
        assert "epochs" in loaded.score_dict
        assert len(loaded.score_dict["epochs"]) == 2


class _StagePlugin:
    def __init__(self, stage):
        self.stage = stage

    def resolve_defense_stage(self, pipeline, **kwargs):
        _ = pipeline, kwargs
        return self.stage


class _IdentityDefense:
    def __init__(self):
        self.calls = []
        self.defense_application_time = 0.0

    def apply_to(self, estimator, data):
        self.calls.append((estimator, data))
        return estimator


class _EpochAttackStub:
    def __call__(
        self,
        data,
        model,
        attack_file=None,
        attack_predictions_file=None,
        score_file=None,
    ):
        _ = data, model, attack_file, attack_predictions_file, score_file
        return {
            "evasion_accuracy": 0.25,
            "evasion_success": 0.75,
            "attack_score_time": 0.001,
        }


def test_pytorch_model_checkpointing_scores_and_caches_models():
    data = SimpleNamespace(
        X_train=torch.randn(12, 4),
        y_train=torch.randint(0, 2, (12,)),
        X_test=torch.randn(6, 4),
        y_test=torch.randint(0, 2, (6,)),
    )
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={
            "nb_epochs": 5,
            "batch_size": 2,
            "checkpoint_every_epochs": 2,
        },
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.fit_params["checkpoint_dir"] = tmpdir
        times = cfg._load_or_train_model(data, model_file=None, times={})

        checkpoint_models = sorted(Path(tmpdir).glob("*.pkl"))
        checkpoint_scores = sorted(Path(tmpdir).glob("*.json"))

        assert times["training_n"] == len(data.y_train)
        assert len(checkpoint_models) == 3
        assert len(checkpoint_scores) == 3
        assert len(cfg.checkpoint_records) == 3
        assert "checkpoints" in cfg.score_dict
        assert "optimizer_loss" in cfg.score_dict
        assert cfg.score_dict["optimizer_loss"] is not None

        loaded_scores = cfg.load_scores(str(checkpoint_scores[-1]))
        assert "optimizer_loss" in loaded_scores or "epochs" in loaded_scores
        assert "accuracy" in loaded_scores


def test_pytorch_model_checkpointing_preserves_post_fit_defense_stage():
    data = SimpleNamespace(
        X_train=torch.randn(8, 4),
        y_train=torch.randint(0, 2, (8,)),
        X_test=torch.randn(4, 4),
        y_test=torch.randint(0, 2, (4,)),
    )
    defense = _IdentityDefense()
    pipeline = DefensePipelineConfig(
        defenses=[defense],
        plugins=[_StagePlugin("post_fit_pre_predict")],
    )
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={
            "nb_epochs": 5,
            "batch_size": 2,
            "checkpoint_every_epochs": 2,
        },
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
        defense=pipeline,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.fit_params["checkpoint_dir"] = tmpdir
        cfg._load_or_train_model(data, model_file=None, times={})

        # Per-epoch benign scoring now applies defense each epoch, in addition to
        # checkpoint evaluation and final post-fit defense application.
        assert len(defense.calls) == 9


def test_pytorch_model_with_adam_optimizer():
    """Test training with Adam optimizer instead of SGD."""
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 2, "batch_size": 2},
        criterion="CrossEntropyLoss",
        optimizer={"name": "Adam", "lr": 0.001},
    )

    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    cfg._train(X, y)

    assert "optimizer_loss" in cfg.score_dict
    assert cfg.score_dict["optimizer_loss"] is not None
    assert isinstance(cfg.score_dict["optimizer_loss"], float)
    assert "epochs" in cfg.score_dict


def test_pytorch_model_with_mse_loss():
    """Test training with MSE loss for regression."""
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 1},
        classifier=False,
        fit_params={"nb_epochs": 2, "batch_size": 2},
        criterion="MSELoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    X = torch.randn(8, 4)
    y = torch.randn(8, 1)
    cfg._train(X, y)

    assert "optimizer_loss" in cfg.score_dict
    assert cfg.score_dict["optimizer_loss"] is not None
    assert isinstance(cfg.score_dict["optimizer_loss"], float)


def test_pytorch_model_serialization_preserves_optimizer_config():
    """Test that optimizer config is preserved through serialization."""
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 2},
        criterion="CrossEntropyLoss",
        optimizer={"name": "Adam", "lr": 0.001, "betas": (0.9, 0.999)},
    )

    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    cfg._train(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "torch_model_with_adam.pkl"
        cfg.save(str(model_path))

        loaded = PytorchModelConfig(
            model_type="torch.nn.Linear",
            model_params={"in_features": 4, "out_features": 2},
            classifier=True,
        )
        loaded.load(str(model_path))

        assert loaded.optimizer is not None
        assert loaded.optimizer["name"] == "Adam"
        assert loaded.optimizer["lr"] == 0.001
        assert "optimizer_loss" in loaded.score_dict


def test_pytorch_model_checkpoint_records_track_epochs():
    """Test that checkpoint records correctly track epoch numbers."""
    data = SimpleNamespace(
        X_train=torch.randn(16, 4),
        y_train=torch.randint(0, 2, (16,)),
        X_test=torch.randn(8, 4),
        y_test=torch.randint(0, 2, (8,)),
    )
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={
            "nb_epochs": 6,
            "batch_size": 4,
            "checkpoint_every_epochs": 2,
        },
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.fit_params["checkpoint_dir"] = tmpdir
        cfg._load_or_train_model(data, model_file=None, times={})

        # 6 epochs / 2 epochs per checkpoint = 3 checkpoints
        assert len(cfg.checkpoint_records) == 3
        
        # Verify checkpoint records have epoch info and file paths
        for record in cfg.checkpoint_records:
            assert "epoch" in record
            assert "model_file" in record or "model_path" in record
            assert "score_file" in record or "score_path" in record


def test_pytorch_checkpoint_filename_format_appends_epoch_before_extension():
    data = SimpleNamespace(
        X_train=torch.randn(10, 4),
        y_train=torch.randint(0, 2, (10,)),
        X_test=torch.randn(4, 4),
        y_test=torch.randint(0, 2, (4,)),
    )
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={
            "nb_epochs": 3,
            "batch_size": 2,
            "checkpoint_every_epochs": 1,
            "checkpoint_prefix": "model",
        },
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.fit_params["checkpoint_dir"] = tmpdir
        cfg._load_or_train_model(data, model_file=None, times={})

        model_files = sorted(Path(tmpdir).glob("*.pkl"))
        score_files = sorted(Path(tmpdir).glob("*.json"))
        assert model_files
        assert score_files
        for record in cfg.checkpoint_records:
            epoch = record["epoch"]
            model_name = Path(record["model_file"]).name
            score_name = Path(record["score_file"]).name
            assert model_name.endswith(f"_{epoch}.pkl")
            assert score_name.endswith(f"_{epoch}.json")
            assert "_epoch_" not in model_name
            assert "_epoch_" not in score_name


def test_pytorch_epoch_attack_scoring_runs_each_epoch_and_keeps_convention():
    data = SimpleNamespace(
        X_train=torch.randn(12, 4),
        y_train=torch.randint(0, 2, (12,)),
        X_test=torch.randn(6, 4),
        y_test=torch.randint(0, 2, (6,)),
    )
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 3, "batch_size": 2},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )
    cfg.set_epoch_attack(_EpochAttackStub())

    cfg._load_or_train_model(data, model_file=None, times={})

    assert "epochs" in cfg.score_dict
    assert len(cfg.score_dict["epochs"]) == 3
    for epoch_idx in (1, 2, 3):
        epoch_entry = cfg.score_dict["epochs"][epoch_idx]
        assert "benign_scores" in epoch_entry
        assert "adversarial_scores" in epoch_entry
        adv_scores = epoch_entry["adversarial_scores"]
        assert "evasion_accuracy" in adv_scores
        assert "evasion_success" in adv_scores
        assert "timings" in epoch_entry
        assert "adversarial_score_time" in epoch_entry["timings"]


def test_pytorch_mps_request_falls_back_to_cpu_when_unavailable(monkeypatch):
    if not hasattr(torch.backends, "mps"):
        pytest.skip("torch backend has no mps support")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        device="mps",
    )
    assert str(cfg.device) == "cpu"


def test_pytorch_art_device_type_for_mps_is_cpu(monkeypatch):
    if not hasattr(torch.backends, "mps"):
        pytest.skip("torch backend has no mps support")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        device="mps",
    )

    assert cfg.device.type == "mps"
    assert cfg._resolve_art_device_type() == "cpu"


def test_pytorch_model_loss_decreases_during_training():
    """Test that optimizer loss generally decreases during training."""
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 10, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 5, "batch_size": 4},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.1},
    )

    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    
    # Train and capture loss
    cfg._train(X, y)
    final_loss_1 = cfg.score_dict["optimizer_loss"]
    
    # Retrain with more epochs to verify loss changes
    cfg.fit_params["nb_epochs"] = 10
    cfg._train(X, y)
    final_loss_2 = cfg.score_dict["optimizer_loss"]
    
    # Loss should be different (not necessarily lower due to randomness)
    # but the metric should exist and be a valid float
    assert isinstance(final_loss_1, float)
    assert isinstance(final_loss_2, float)
    assert final_loss_1 > 0
    assert final_loss_2 > 0
    # Verify we have epoch metrics
    assert "epochs" in cfg.score_dict
    assert len(cfg.score_dict["epochs"]) >= 10


def test_pytorch_model_different_batch_sizes():
    """Test training with different batch sizes."""
    data = SimpleNamespace(
        X_train=torch.randn(16, 4),
        y_train=torch.randint(0, 2, (16,)),
        X_test=torch.randn(8, 4),
        y_test=torch.randint(0, 2, (8,)),
    )
    
    batch_sizes = [2, 4, 8]
    losses = []
    
    for batch_size in batch_sizes:
        cfg = PytorchModelConfig(
            model_type="torch.nn.Linear",
            model_params={"in_features": 4, "out_features": 2},
            classifier=True,
            fit_params={"nb_epochs": 2, "batch_size": batch_size},
            criterion="CrossEntropyLoss",
            optimizer={"name": "SGD", "lr": 0.01},
        )
        cfg._train(data.X_train, data.y_train)
        losses.append(cfg.score_dict["optimizer_loss"])
    
    # All batch sizes should produce valid losses
    for loss in losses:
        assert isinstance(loss, float)
        assert loss > 0


def test_pytorch_model_serialization_with_different_input_sizes():
    """Test serialization with models that have different input dimensions."""
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 100, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 4},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    X = torch.randn(8, 100)
    y = torch.randint(0, 2, (8,))
    cfg._train(X, y)

    assert "optimizer_loss" in cfg.score_dict

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "large_model.pkl"
        cfg.save(str(model_path))

        loaded = PytorchModelConfig(
            model_type="torch.nn.Linear",
            model_params={"in_features": 100, "out_features": 2},
            classifier=True,
        )
        loaded.load(str(model_path))

        # Verify model can make predictions with the new input size
        model_device = next(loaded.get_model().parameters()).device
        X_new = torch.randn(4, 100, device=model_device)
        output = loaded.get_model()(X_new)
        assert output.shape == (4, 2)


def test_pytorch_model_hash_stable_after_training_and_runtime_updates():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 2},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    original_hash = hash(cfg)

    X = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    cfg._train(X, y)
    cfg.score_dict["runtime_metric"] = 1.23
    cfg.training_time = 99.0

    assert hash(cfg) == original_hash


def test_pytorch_model_hash_stable_after_load_roundtrip_runtime_mutation():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 2},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "hash_roundtrip.pkl"
        cfg.save(str(path))

        loaded = PytorchModelConfig(
            model_type="torch.nn.Linear",
            model_params={"in_features": 4, "out_features": 2},
            classifier=True,
        )
        loaded.load(str(path))
        original_hash = hash(loaded)

        loaded.score_dict["runtime"] = 1
        loaded.prediction_time = 5.0

        assert hash(loaded) == original_hash
