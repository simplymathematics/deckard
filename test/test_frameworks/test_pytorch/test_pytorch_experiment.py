"""Comprehensive tests for PyTorch-based experiments.

Working directory context: examples/pytorch (uses torch_default.yaml via .deckard_rc).
Covers: unit, integration, hash stability, persistence, ordering, device propagation, and subcommand tests.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from helpers import make_runtime_env

from deckard.attack import AttackConfig
from deckard.experiment import TorchExperimentConfig
from deckard.file import FileConfig
from deckard.frameworks.pytorch.data import PytorchDataConfig
from deckard.frameworks.pytorch.model import PytorchModelConfig
from deckard.model import DefensePipelineConfig
from deckard.model.defend import DefenseConfig

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_PYTORCH_DIR = ROOT / "examples" / "pytorch"
DECKARD_RC_PATH = EXAMPLES_PYTORCH_DIR / ".deckard_rc"


def _runtime_env() -> dict[str, str]:
    env = make_runtime_env(DECKARD_RC_PATH)
    env["DECKARD_DEFAULT_CONFIG_FILE"] = "torch_default_cli.yaml"
    env["DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION"] = "1"
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_torch_data(unit_interval: bool = False):
    X = torch.rand(60, 8) if unit_interval else torch.randn(60, 8)
    y = torch.randint(0, 2, (60,))
    data = PytorchDataConfig(
        dataset_name="torch.utils.data.TensorDataset",
        train_size=40,
        test_size=20,
        classifier=True,
        random_state=42,
        data_params={"_args_": [X, y]},
    )
    data()
    return data


def _make_unloaded_torch_data():
    X = torch.randn(40, 8)
    y = torch.randint(0, 2, (40,))
    return PytorchDataConfig(
        dataset_name="torch.utils.data.TensorDataset",
        train_size=30,
        test_size=10,
        classifier=True,
        data_params={"_args_": [X, y]},
    )


def _make_torch_model(defense=None):
    return PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 8, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 16},
        criterion="CrossEntropyLoss",
        optimizer={"name": "SGD", "lr": 0.01},
        defense=defense,
    )


def _make_torch_fairlearn_defense_from_yaml(yaml_file: str):
    cfg_path = EXAMPLES_PYTORCH_DIR / "config" / "defense" / yaml_file
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return {
        "defense_name": cfg["defense_name"],
        "defense_params": cfg["defense_params"],
        "classifier": True,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_pytorch_model_config_initialises():
    model = _make_torch_model()
    assert model.classifier is True
    assert model.model_type == "torch.nn.Linear"


def test_pytorch_model_config_hash_stable_before_training():
    model = _make_torch_model()
    assert hash(model) == hash(model)


def test_torch_experiment_canonical_device_handles_null_aliases():
    assert TorchExperimentConfig._canonical_device(None) is None
    assert TorchExperimentConfig._canonical_device("") is None
    assert TorchExperimentConfig._canonical_device(" auto ") is None
    assert TorchExperimentConfig._canonical_device("DEFAULT") is None
    assert TorchExperimentConfig._canonical_device("CuDa:0") == "cuda:0"


def test_pytorch_defense_pipeline_initialises():
    pipeline = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": False, "apply_predict": True},
                "classifier": True,
            },
        ],
    )
    assert len(pipeline.defenses) == 1
    assert isinstance(pipeline.defenses[0], DefenseConfig)


def test_pytorch_art_defense_is_detected_as_art_by_pipeline():
    pipeline = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": False, "apply_predict": True},
                "classifier": True,
            },
        ],
    )
    assert pipeline._is_art_defense(pipeline.defenses[0])


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_pytorch_model_trains_on_synthetic_data():
    data = _make_torch_data()
    model = _make_torch_model()
    model(data)
    assert "accuracy" in model.score_dict
    assert "training_time" in model.score_dict


def test_pytorch_model_with_art_postprocessor_defense():
    data = _make_torch_data()
    defense_cfg = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": False, "apply_predict": True},
                "classifier": True,
            },
        ],
    )
    model = _make_torch_model(defense=defense_cfg)
    model(data)
    assert "accuracy" in model.score_dict
    assert model.defense_application_time is not None


def test_pytorch_model_with_art_preprocessor_defense():
    data = _make_torch_data(unit_interval=True)
    defense_cfg = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.preprocessor.FeatureSqueezing",
                "defense_params": {"clip_values": [0.0, 1.0], "bit_depth": 4},
                "classifier": True,
            },
        ],
    )
    model = _make_torch_model(defense=defense_cfg)
    model(data)
    assert "accuracy" in model.score_dict


def test_pytorch_experiment_end_to_end():
    exp = TorchExperimentConfig(
        data=_make_torch_data(),
        model=_make_torch_model(),
        attack=None,
        files=FileConfig(),
        classifier=True,
    )
    scores = exp()
    assert isinstance(scores, dict)
    assert "accuracy" in scores


# ---------------------------------------------------------------------------
# Device propagation tests moved from generic experiment suite
# ---------------------------------------------------------------------------


def test_pytorch_device_propagates_from_experiment():
    data = _make_unloaded_torch_data()
    model = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 8, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 8},
    )

    exp = TorchExperimentConfig(
        data=data,
        model=model,
        attack=None,
        files=FileConfig(),
        device="cpu",
        classifier=True,
    )

    expected = str(getattr(exp.data, "device"))
    assert str(exp.device) == expected
    assert str(getattr(exp.model, "device")) == expected


def test_pytorch_device_propagates_from_model_when_only_one_specified():
    data = _make_unloaded_torch_data()
    model = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 8, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 8},
    )
    data.device = None
    model.device = "cpu"

    exp = TorchExperimentConfig(
        data=data,
        model=model,
        attack=None,
        files=FileConfig(),
        classifier=True,
    )

    expected = str(getattr(exp.data, "device"))
    assert str(exp.device) == expected
    assert str(getattr(exp.model, "device")) == expected


def test_pytorch_auto_device_propagates_to_all_components():
    data = _make_unloaded_torch_data()
    model = PytorchModelConfig(
        model_type="torch.nn.Linear",
        model_params={"in_features": 8, "out_features": 2},
        classifier=True,
        fit_params={"nb_epochs": 1, "batch_size": 8},
    )

    with (
        patch(
            "deckard.frameworks.pytorch.experiment.resolve_torch_device",
        ) as resolve_device,
        patch(
            "deckard.attack.base.resolve_torch_device",
        ) as attack_resolve_device,
    ):
        resolve_device.return_value = "cpu"
        attack_resolve_device.return_value = "cpu"
        attack = AttackConfig(attack_size=1)
        exp = TorchExperimentConfig(
            data=data,
            model=model,
            attack=attack,
            files=FileConfig(),
            classifier=True,
        )

    assert str(exp.device) == "cpu"
    assert str(getattr(exp.data, "device")) == "cpu"
    assert str(getattr(exp.model, "device")) == "cpu"
    assert str(getattr(exp.attack, "device")) == "cpu"


# ---------------------------------------------------------------------------
# Ordering tests
# ---------------------------------------------------------------------------


def test_art_last_ordering_no_warning_for_wrapper_only_chain(caplog):
    import logging

    fairlearn_defense = _make_torch_fairlearn_defense_from_yaml(
        yaml_file="fairlearn-adversarial-classifier.yaml",
    )
    art_defense_dict = {
        "defense_name": "art.defences.postprocessor.ClassLabels",
        "defense_params": {"apply_fit": False, "apply_predict": True},
        "classifier": True,
    }

    pipeline = DefensePipelineConfig(
        defenses=[art_defense_dict, fairlearn_defense],
    )

    with caplog.at_level(logging.WARNING, logger="deckard.model.defend"):
        data_cfg = _make_torch_data(unit_interval=True)
        estimator = _make_torch_model()._model
        try:
            pipeline.apply(estimator=estimator, data=data_cfg)
        except Exception:
            pass

    reorder_warnings = [
        r for r in caplog.records if "automatically reordered" in r.message
    ]
    assert len(reorder_warnings) == 0


def test_art_last_ordering_no_warning_when_already_last(caplog):
    import logging

    pipeline = DefensePipelineConfig(
        defenses=[
            {
                "defense_name": "art.defences.postprocessor.ClassLabels",
                "defense_params": {"apply_fit": False, "apply_predict": True},
                "classifier": True,
            },
        ],
    )

    with caplog.at_level(logging.WARNING, logger="deckard.model.defend"):
        data_cfg = _make_torch_data(unit_interval=True)
        estimator = _make_torch_model()._model
        try:
            pipeline.apply(estimator=estimator, data=data_cfg)
        except Exception:
            pass

    reorder_warnings = [r for r in caplog.records if "reorder" in r.message.lower()]
    assert len(reorder_warnings) == 0


# ---------------------------------------------------------------------------
# Hash stability tests
# ---------------------------------------------------------------------------


def test_pytorch_model_hash_stable_after_training():
    data = _make_torch_data()
    model = _make_torch_model()
    h_before = hash(model)
    model(data)
    assert h_before == hash(model)


def test_torch_experiment_coerces_data_batch_size_from_model_fit_params():
    data_cfg = _make_torch_data()
    data_cfg.data_params.pop("batch_size", None)
    model_cfg = _make_torch_model()
    model_cfg.fit_params["batch_size"] = 16

    exp = TorchExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        attack=None,
        files=FileConfig(),
        classifier=True,
    )

    assert exp.data.data_params.get("batch_size") == 16
    assert exp.model.fit_params.get("batch_size") == 16


def test_torch_experiment_raises_on_batch_size_mismatch():
    data_cfg = _make_torch_data()
    data_cfg.data_params["batch_size"] = 8
    model_cfg = _make_torch_model()
    model_cfg.fit_params["batch_size"] = 16

    with pytest.raises(ValueError, match="batch_size"):
        TorchExperimentConfig(
            data=data_cfg,
            model=model_cfg,
            attack=None,
            files=FileConfig(),
            classifier=True,
        )


def test_torch_experiment_reconcile_component_devices_raises_on_conflict():
    exp = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp.device = "cuda:0"
    exp.data = SimpleNamespace(device="mps")
    exp.model = SimpleNamespace(device="cuda:0")
    exp.attack = None

    with pytest.raises(AssertionError, match="must match"):
        exp._reconcile_component_devices()


def test_torch_experiment_reconcile_component_devices_prefers_model_then_attack_then_data(
    monkeypatch,
):
    monkeypatch.setattr(
        "deckard.frameworks.pytorch.experiment.resolve_torch_device",
        lambda _d=None: "cpu",
    )
    exp = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp.device = None
    exp.data = SimpleNamespace(device="cpu")
    exp.model = SimpleNamespace(device="cuda:1")
    exp.attack = None
    exp._reconcile_component_devices()
    assert exp.device == "cuda:1"

    exp2 = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp2.device = None
    exp2.data = SimpleNamespace(device="cpu")
    exp2.model = None
    exp2.attack = SimpleNamespace(device="cuda:2")
    exp2._reconcile_component_devices()
    assert exp2.device == "cuda:2"

    exp3 = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp3.device = None
    exp3.data = SimpleNamespace(device="mps")
    exp3.model = None
    exp3.attack = None
    exp3._reconcile_component_devices()
    assert exp3.device == "mps"


def test_torch_experiment_reconcile_component_devices_uses_resolver_fallback(
    monkeypatch,
):
    monkeypatch.setattr(
        "deckard.frameworks.pytorch.experiment.resolve_torch_device",
        lambda _d=None: "cpu",
    )
    exp = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp.device = None
    exp.data = SimpleNamespace(device=None)
    exp.model = None
    exp.attack = None

    exp._reconcile_component_devices()
    assert exp.device == "cpu"


def test_torch_experiment_reconcile_component_devices_detects_post_resolution_mismatch():
    exp = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp.device = "cpu"
    exp.data = SimpleNamespace(device="cpu")

    class _Model:
        device = "cpu"

        @staticmethod
        def _resolve_torch_device(_device):
            return "cuda:0"

    exp.model = _Model()
    exp.attack = None

    with pytest.raises(AssertionError, match="identical after reconciliation"):
        exp._reconcile_component_devices()


def test_torch_experiment_reconcile_batch_size_handles_model_none_and_missing_params():
    exp = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp.model = None
    exp.data = SimpleNamespace(data_params=None)
    exp._reconcile_batch_size()

    exp2 = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp2.model = SimpleNamespace(fit_params=None)
    exp2.data = SimpleNamespace(data_params=None)
    exp2._reconcile_batch_size()
    assert exp2.model.fit_params == {}
    assert exp2.data.data_params == {}


def test_torch_experiment_reconcile_batch_size_copies_from_data_to_model():
    exp = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp.model = SimpleNamespace(fit_params={})
    exp.data = SimpleNamespace(data_params={"batch_size": 32})
    exp._reconcile_batch_size()
    assert exp.model.fit_params["batch_size"] == 32


def test_torch_experiment_type_enforcement_errors():
    exp = TorchExperimentConfig.__new__(TorchExperimentConfig)
    exp.data = object()
    exp.model = object()

    with pytest.raises(TypeError, match="requires data to be a PytorchDataConfig"):
        exp._enforce_torch_data()

    with pytest.raises(TypeError, match="requires model to be a PytorchModelConfig"):
        exp._enforce_torch_model()


def test_torch_experiment_post_init_rejects_non_pytorch_library():
    with pytest.raises(ValueError, match="must use library='pytorch'"):
        TorchExperimentConfig(
            data=_make_torch_data(),
            model=_make_torch_model(),
            attack=None,
            files=FileConfig(),
            classifier=True,
            library="sklearn",
        )


def test_pytorch_experiment_hash_stable_after_execution():
    exp = TorchExperimentConfig(
        data=_make_torch_data(),
        model=_make_torch_model(),
        attack=None,
        files=FileConfig(),
        classifier=True,
    )
    h_before = hash(exp)
    exp()
    assert h_before == hash(exp)


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_pytorch_scores_persist_to_json():
    data = _make_torch_data()
    model = _make_torch_model()
    model(data)

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
        mode="w",
    ) as f:
        score_path = f.name
        json.dump(model.score_dict, f)

    try:
        with open(score_path) as f:
            loaded = json.load(f)
        assert "accuracy" in loaded
        assert loaded["accuracy"] == pytest.approx(
            model.score_dict["accuracy"],
            abs=1e-9,
        )
    finally:
        os.unlink(score_path)


def test_pytorch_experiment_scores_persist_via_file_config(tmp_path):
    score_file = str(tmp_path / "scores.json")
    exp = TorchExperimentConfig(
        data=_make_torch_data(),
        model=_make_torch_model(),
        attack=None,
        files=FileConfig(score_file=score_file),
        classifier=True,
    )
    exp()

    assert Path(score_file).exists()
    with open(score_file) as f:
        loaded = json.load(f)
    assert "accuracy" in loaded


# ---------------------------------------------------------------------------
# Subcommand test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/pytorch/.deckard_rc not found",
)
def test_deckard_optimize_subcommand_help_in_pytorch_dir():
    env = _runtime_env()

    result = subprocess.run(
        [sys.executable, "-m", "deckard", "optimize", "--help"],
        cwd=str(EXAMPLES_PYTORCH_DIR),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, f"deckard optimize --help failed:\n{result.stderr}"


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/pytorch/.deckard_rc not found",
)
def test_deckard_optimize_help_in_examples_pytorch():
    result = subprocess.run(
        [sys.executable, "-m", "deckard", "optimize", "--help"],
        cwd=str(EXAMPLES_PYTORCH_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/pytorch/.deckard_rc not found",
)
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torchvision") is None,
    reason="torchvision not installed",
)
def test_deckard_optimize_torch_art_smoke_matrix():
    cmd = [
        sys.executable,
        "-m",
        "deckard",
        "optimize",
        "data=torch_mnist",
        "model=default",
        "attack=fgm",
        "+defense=class_labels",
        "experiment_name=torch_art_smoke_chain",
        "files.model_file=null",
        "files.attack_file=null",
        "data.train_size=64",
        "data.test_size=32",
        "model.fit_params.nb_epochs=1",
        "model.fit_params.batch_size=64",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(EXAMPLES_PYTORCH_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/pytorch/.deckard_rc not found",
)
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torchvision") is None,
    reason="torchvision not installed",
)
def test_deckard_optimize_torch_poisoning_gradient_matching_smoke_matrix():
    cmd = [
        sys.executable,
        "-m",
        "deckard",
        "optimize",
        "data=torch_mnist",
        "model=default",
        "attack=poisoning-gradient-matching",
        "experiment_name=torch_poisoning_smoke_chain",
        "files.model_file=null",
        "files.attack_file=null",
        "data.train_size=64",
        "data.test_size=32",
        "+data.data_params.num_workers=0",
        "+model.device=cpu",
        "+attack.device=cpu",
        "model.fit_params.nb_epochs=1",
        "model.fit_params.batch_size=32",
        "attack.attack_params.percent_poison=0.05",
        "attack.attack_params.max_epochs=1",
        "attack.attack_params.max_trials=1",
        "attack.attack_params.class_source=0",
        "attack.attack_params.class_target=1",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(EXAMPLES_PYTORCH_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0


@pytest.mark.skipif(
    not DECKARD_RC_PATH.exists(),
    reason="examples/pytorch/.deckard_rc not found",
)
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("fairlearn") is None,
    reason="fairlearn is required for torch fairness smoke",
)
def test_deckard_optimize_torch_fairness_smoke_matrix():
    cmd = [
        sys.executable,
        "-m",
        "deckard",
        "optimize",
        "data=fairlearn_celeba",
        "model=default",
        "~attack",
        "+defense=fairlearn-adversarial-classifier",
        "experiment_name=torch_fairness_smoke_chain",
        "files.model_file=null",
        "files.attack_file=null",
        "data.train_size=64",
        "data.test_size=32",
        "data.dataset_name=torch_fairness_dataset.py:SyntheticTabularFairnessDataset",
        "+data.data_params.num_samples=200",
        "+data.data_params.n_features=16",
        "model.model_type=torch.nn.Linear",
        "~model.model_params.num_channels",
        "~model.model_params.num_classes",
        "+model.model_params={in_features:16,out_features:2}",
        "model.fit_params.nb_epochs=1",
        "model.fit_params.batch_size=32",
        "+model.device=cpu",
        "defense.defense_params.epochs=1",
        "defense.defense_params.batch_size=16",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(EXAMPLES_PYTORCH_DIR),
        env=_runtime_env(),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0
