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
from unittest.mock import patch

import pytest
import yaml


from deckard.attack import AttackConfig
from deckard.data.pytorch import PytorchDataConfig
from deckard.experiment import TorchExperimentConfig
from deckard.file import FileConfig
from deckard.model import DefensePipelineConfig
from deckard.model.defend import DefenseConfig
from deckard.model.pytorch import PytorchModelConfig

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PYTORCH_DIR = ROOT / "examples" / "pytorch"
DECKARD_RC_PATH = EXAMPLES_PYTORCH_DIR / ".deckard_rc"

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
            "deckard.experiment.torch_experiment.resolve_torch_device",
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
    env = os.environ.copy()
    for raw_line in DECKARD_RC_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        kv = line[len("export ") :]
        if "=" not in kv:
            continue
        key, value = kv.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    env["DECKARD_TEST_MAX_SAMPLES"] = "200"

    result = subprocess.run(
        [sys.executable, "-m", "deckard", "optimize", "--help"],
        cwd=str(EXAMPLES_PYTORCH_DIR),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, f"deckard optimize --help failed:\n{result.stderr}"


def _load_env_from_deckard_rc(path: Path) -> dict[str, str]:
    env_overrides: dict[str, str] = {}
    if not path.exists():
        return env_overrides
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        key_value = line[len("export ") :]
        if "=" not in key_value:
            continue
        key, value = key_value.split("=", 1)
        env_overrides[key.strip()] = value.strip().strip('"').strip("'")
    return env_overrides


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(_load_env_from_deckard_rc(DECKARD_RC_PATH))
    env["DECKARD_TEST_MAX_SAMPLES"] = "200"
    env.setdefault("MPLBACKEND", "Agg")
    return env


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
    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


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
    assert (
        result.returncode == 0
    ), f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
