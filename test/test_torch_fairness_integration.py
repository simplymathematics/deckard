"""Runtime integration tests for torch fairness and ART chains in examples/pytorch."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PYTORCH_DIR = ROOT / "examples" / "pytorch"
DECKARD_RC_PATH = EXAMPLES_PYTORCH_DIR / ".deckard_rc"


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
