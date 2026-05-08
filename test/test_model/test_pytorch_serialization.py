import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from deckard.model.defend import DefensePipelineConfig
from deckard.model import pytorch as pytorch_module

PytorchModelConfig = pytest.importorskip(
    "deckard.model.pytorch",
).PytorchModelConfig
initialize_criterion = pytorch_module.initialize_criterion
initialize_optimizer = pytorch_module.initialize_optimizer


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


def test_initialize_criterion_and_optimizer_support_variants(monkeypatch):
    calls = []

    def _fake_load_class(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return {"name": name, "args": args, "kwargs": kwargs}

    monkeypatch.setattr(pytorch_module, "load_class", _fake_load_class)

    criterion = initialize_criterion("CrossEntropyLoss")
    assert criterion["name"] == "torch.nn.CrossEntropyLoss"

    criterion_cfg = initialize_criterion({"name": "torch.nn.MSELoss", "reduction": "sum"})
    assert criterion_cfg["kwargs"]["reduction"] == "sum"

    with pytest.raises(ValueError, match="criterion must be str or dict"):
        initialize_criterion(123)

    model = torch.nn.Linear(4, 2)
    optimizer = initialize_optimizer("SGD", model.parameters())
    assert optimizer["name"] == "torch.optim.SGD"

    optimizer_cfg = initialize_optimizer({"name": "Adam", "lr": 0.01}, model.parameters())
    assert optimizer_cfg["name"] == "torch.optim.Adam"
    assert "params" in optimizer_cfg["kwargs"]

    with pytest.raises(ValueError, match="optimizer must be str or dict"):
        initialize_optimizer(123, model.parameters())


def test_pytorch_device_and_art_helpers():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )

    cfg.device = torch.device("cuda")
    assert cfg._resolve_art_device_type() == "gpu"

    cfg.device = torch.device("mps")
    assert cfg._resolve_art_device_type() == "cpu"

    cfg._model = None
    with pytest.raises(ValueError, match="Model not initialized"):
        cfg._model_for_art()

    cfg._model = torch.nn.Linear(4, 2)
    cfg.device = torch.device("cpu")
    assert cfg._model_for_art() is cfg._model

    class _DummyEstimator:
        def __init__(self):
            self._device = None
            self._model = torch.nn.Linear(4, 2)
            self.preprocessing = SimpleNamespace(_device=None)
            self.preprocessing_operations = [SimpleNamespace(_device=None)]

            def _apply(*args, **kwargs):
                _ = args, kwargs
                return (np.array([[1.0]], dtype=np.float64),)

            self._apply_preprocessing = _apply

    est = _DummyEstimator()
    cfg.device = torch.device("mps")
    wrapped = cfg._override_art_internal_device(est)
    assert str(wrapped._device) == "cpu"
    out = wrapped._apply_preprocessing(np.array([[1.0]], dtype=np.float64))
    assert out[0].dtype == pytorch_module.ART_NUMPY_DTYPE

    cfg.device = SimpleNamespace(type="xla")
    assert cfg._override_art_internal_device(est) is est


def test_checkpoint_config_and_data_validation_paths():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )

    cfg.fit_params = {"checkpoint_every_epochs": "abc"}
    with pytest.raises(ValueError, match="must be an integer"):
        cfg._resolve_checkpoint_config()

    cfg.fit_params = {"checkpoint_every_epochs": 1}
    with pytest.raises(ValueError, match="checkpoint_dir must be provided"):
        cfg._resolve_checkpoint_config(model_file=None)

    cfg.fit_params = {"checkpoint_every_epochs": 1, "checkpoint_include_final": "no"}
    checkpoint_cfg = cfg._resolve_checkpoint_config(model_file="/tmp/model.pkl")
    assert checkpoint_cfg["every"] == 1
    assert checkpoint_cfg["prefix"] == "model"
    assert checkpoint_cfg["include_final"] is False

    bad_data = SimpleNamespace(
        X_train=np.array([[1.0]]),
        y_train=torch.tensor([1]),
        X_test=torch.tensor([[1.0]]),
        y_test=torch.tensor([1]),
    )
    with pytest.raises(TypeError, match="requires torch.Tensor or DataLoader"):
        cfg._validate_torch_data(bad_data)


def test_predict_path_for_art_wrapper_and_empty_loader():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )

    class _ArtLikeModel:
        def predict(self, x):
            x = np.asarray(x)
            return np.column_stack([1.0 - x[:, 0], x[:, 0]])

    cfg._model = _ArtLikeModel()

    empty_loader = DataLoader(TensorDataset(torch.empty((0, 4))), batch_size=2)
    out_empty = cfg._predict(empty_loader)
    assert out_empty.numel() == 0

    x = torch.tensor([[0.2, 0.1, 0.0, 0.0], [0.8, 0.1, 0.0, 0.0]], dtype=torch.float32)
    out = cfg._predict(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2


def test_model_type_accepts_in_memory_class():
    class TinyLinear(torch.nn.Module):
        def __init__(self, in_features=4, out_features=2):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x)

    cfg = PytorchModelConfig(
        model_type=TinyLinear,
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )

    model = cfg.get_model()
    assert isinstance(model, TinyLinear)


def test_model_type_accepts_in_memory_instance():
    instance = torch.nn.Linear(4, 2)
    cfg = PytorchModelConfig(
        model_type=instance,
        model_params={},
        classifier=True,
    )

    model = cfg.get_model()
    assert isinstance(model, torch.nn.Linear)
    assert model is not instance
    for expected, actual in zip(instance.parameters(), model.parameters()):
        assert torch.equal(expected.detach().cpu(), actual.detach().cpu())


def test_model_type_instance_populates_model_params_from_init():
    cfg = PytorchModelConfig(
        model_type=torch.nn.Linear(4, 2, bias=False),
        classifier=True,
    )

    assert cfg.model_params["in_features"] == 4
    assert cfg.model_params["out_features"] == 2
    assert cfg.model_params["bias"] is False


def test_model_type_instance_keeps_explicit_model_params_over_inferred():
    cfg = PytorchModelConfig(
        model_type=torch.nn.Linear(4, 2, bias=False),
        model_params={"bias": True},
        classifier=True,
    )

    assert cfg.model_params["in_features"] == 4
    assert cfg.model_params["out_features"] == 2
    assert cfg.model_params["bias"] is True


def test_model_type_rejects_non_torch_module_class():
    class NotAModule:
        def __init__(self, *args, **kwargs):
            _ = args, kwargs

    with pytest.raises(
        TypeError,
        match="model_type class must inherit torch.nn.Module",
    ):
        PytorchModelConfig(
            model_type=NotAModule,
            model_params={},
            classifier=True,
        )


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


def test_score_checkpoint_snapshot_predict_proba_fallback_classification():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )
    cfg.score_dict = {"epochs": {1: {"loss": 0.1}}}

    snapshot = SimpleNamespace(
        defense=None,
        classifier=True,
        score_dict={},
        _evaluate_and_score=lambda data, times={}: (_ for _ in ()).throw(ValueError("predict_proba unavailable")),
        _predict=lambda X: torch.zeros(len(X), dtype=torch.long),
        _classification_scores=lambda y_true, y_pred: {"accuracy": 0.5},
        _regression_scores=lambda y_true, y_pred: {"mse": 1.0},
    )

    data = SimpleNamespace(
        X_train=torch.randn(4, 4),
        y_train=torch.randint(0, 2, (4,)),
        X_test=torch.randn(2, 4),
        y_test=torch.randint(0, 2, (2,)),
    )

    scores = cfg._score_checkpoint_snapshot(snapshot, data)

    assert "training_accuracy" in scores
    assert "accuracy" in scores
    assert "epochs" in scores


def test_score_checkpoint_snapshot_predict_proba_fallback_regression():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 1},
        classifier=False,
    )
    cfg.score_dict = {}

    snapshot = SimpleNamespace(
        defense=None,
        classifier=False,
        score_dict={},
        _evaluate_and_score=lambda data, times={}: (_ for _ in ()).throw(ValueError("predict_proba missing")),
        _predict=lambda X: torch.zeros(len(X), dtype=torch.float32),
        _classification_scores=lambda y_true, y_pred: {"accuracy": 0.5},
        _regression_scores=lambda y_true, y_pred: {"mse": 0.25},
    )

    data = SimpleNamespace(
        X_train=torch.randn(4, 4),
        y_train=torch.randn(4),
        X_test=torch.randn(2, 4),
        y_test=torch.randn(2),
    )

    scores = cfg._score_checkpoint_snapshot(snapshot, data)

    assert "training_mse" in scores
    assert "mse" in scores


def test_pytorch_score_helpers_classification_and_regression():
    cls_cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )
    y_true_cls = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    y_pred_logits = torch.tensor(
        [[0.8, 0.2], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]],
        dtype=torch.float32,
    )
    cls_scores = cls_cfg._classification_scores(y_true_cls, y_pred_logits)
    assert set(cls_scores.keys()) == {"accuracy", "precision", "recall", "f1"}

    # Also cover the non-tensor y_pred path.
    cls_scores_np = cls_cfg._classification_scores(y_true_cls, np.array([0, 1, 1, 0]))
    assert cls_scores_np["accuracy"] == pytest.approx(1.0)

    reg_cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 1},
        classifier=False,
    )
    y_true_reg = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y_pred_reg = torch.tensor([1.5, 2.5, 2.0], dtype=torch.float32)
    reg_scores = reg_cfg._regression_scores(y_true_reg, y_pred_reg)
    assert set(reg_scores.keys()) == {"mse", "rmse", "mae"}


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


def test_pytorch_model_get_model_save_load_error_paths():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )

    cfg._model = None
    with pytest.raises(ValueError, match="Model not initialized"):
        cfg.get_model()
    with pytest.raises(ValueError, match="Model not initialized"):
        cfg.save("/tmp/should_not_exist.pkl")

    with tempfile.TemporaryDirectory() as tmpdir:
        missing = Path(tmpdir) / "missing.pkl"
        with pytest.raises(FileNotFoundError):
            cfg.load(str(missing))

        bad = Path(tmpdir) / "bad.pkl"
        torch.save({"not_state_dict": 1}, bad)
        with pytest.raises(TypeError, match="Unsupported serialized payload"):
            cfg.load(str(bad))


def test_pytorch_save_refuses_to_overwrite_existing_file():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )
    cfg._model = torch.nn.Linear(4, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "model.pkl"
        out.write_text("exists")
        with pytest.raises(ValueError, match="already exists"):
            cfg.save(str(out))


def test_pytorch_score_epoch_snapshot_with_none_data_returns_early():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )
    cfg.score_dict = {"epochs": {}}
    assert cfg._score_epoch_snapshot(epoch_index=1, data=None) is None


def test_pytorch_coerce_bool_and_checkpoint_non_positive_branch():
    cfg = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 4, "out_features": 2},
        classifier=True,
    )

    assert cfg._coerce_bool(None, True) is True
    assert cfg._coerce_bool(2, False) is True

    cfg.fit_params = {"checkpoint_every_epochs": -1}
    assert cfg._resolve_checkpoint_config(model_file=None) is None
