import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from helpers import load_env_from_deckard_rc, make_runtime_env


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_SKLEARN_DIR = ROOT / "examples" / "sklearn"
DECKARD_RC_PATH = EXAMPLES_SKLEARN_DIR / ".deckard_rc"


DURATION_COL_BY_DATASET = {
    "diabetes": "right",
    "leukemia": "t",
    "lung": "time",
    "lifelines_diabetes": "right",
}


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lifelines") is None,
    reason="lifelines is required for survival integration tests",
)
@pytest.mark.parametrize(
    "dataset_name,survival_model",
    [
        ("diabetes", "weibull"),
        ("leukemia", "weibull"),
        ("lung", "cox"),
        ("lifelines_diabetes", "weibull"),
        ("lifelines_diabetes", "cox"),
    ],
)
def test_survival_cli_in_examples_sklearn(
    dataset_name,
    survival_model,
    tmp_path,
):
    examples_dir = EXAMPLES_SKLEARN_DIR
    env = make_runtime_env(DECKARD_RC_PATH)
    env["DECKARD_DEFAULT_CONFIG_FILE"] = "survival.yaml"
    duration_col = DURATION_COL_BY_DATASET[dataset_name]

    deckard_cli = shutil.which("deckard")
    if deckard_cli is not None:
        probe = subprocess.run(
            [deckard_cli, "--help"],
            env=env,
            cwd=examples_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        probe = None

    if probe is not None and probe.returncode == 0:
        cmd = [
            deckard_cli,
            "survival",
            f"data={dataset_name}",
            f"duration_col={duration_col}",
            f"model={survival_model}",
            "score=survival",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "deckard",
            "survival",
            f"data={dataset_name}",
            f"duration_col={duration_col}",
            f"model={survival_model}",
            "score=survival",
        ]

    completed = subprocess.run(
        cmd,
        cwd=examples_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )

    expected_plot = examples_dir / "plots" / "survival" / f"{survival_model}_aft.pdf"
    expected_table = examples_dir / "plots" / "survival" / "aft_comparison.csv"
    assert expected_plot.exists()
    assert expected_table.exists()

    # The survival score profile now includes information-criterion metrics.
    with expected_table.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert "AIC" in header


# ---------------------------------------------------------------------------
# Hash stability and persistence (Python API level, no lifelines runtime needed)
# ---------------------------------------------------------------------------

lifelines_installed = __import__("importlib").util.find_spec("lifelines") is not None


@pytest.mark.skipif(
    not lifelines_installed,
    reason="lifelines is required for LifelinesDataConfig hash tests",
)
def test_lifelines_data_config_hash_stable_after_execution():
    from deckard.data.survival import LifelinesDataConfig, LifelinesDataMode

    cfg = LifelinesDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 6,
            "n_informative": 3,
            "n_redundant": 0,
            "random_state": 7,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
        mode=LifelinesDataMode.AUXILIARY_MODEL,
        duration_col="T",
        event_col="E",
        benign_metric="accuracy",
    )
    original_hash = hash(cfg)
    cfg.score_dict["runtime_metric"] = 1.0
    assert hash(cfg) == original_hash


@pytest.mark.skipif(
    not lifelines_installed,
    reason="lifelines is required for LifelinesDataConfig persistence tests",
)
def test_lifelines_data_config_scores_persist_and_reload():
    from deckard.data.survival import LifelinesDataConfig, LifelinesDataMode

    cfg = LifelinesDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 6,
            "n_informative": 3,
            "n_redundant": 0,
            "random_state": 11,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
        mode=LifelinesDataMode.AUXILIARY_MODEL,
        duration_col="T",
        event_col="E",
        benign_metric="accuracy",
    )
    scores = {"concordance": 0.71, "aic": 120.5}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "survival_scores.json"
        cfg.save_scores(scores, path)
        loaded = cfg.load_scores(str(path))
    assert loaded["concordance"] == pytest.approx(0.71)
    assert "aic" in loaded


@pytest.mark.skipif(
    not lifelines_installed,
    reason="lifelines is required for LifelinesDataConfig pickle tests",
)
def test_lifelines_data_config_object_pickle_roundtrip():
    from deckard.data.survival import LifelinesDataConfig, LifelinesDataMode

    cfg = LifelinesDataConfig(
        dataset_name="make_classification",
        data_params={
            "n_samples": 40,
            "n_features": 6,
            "n_informative": 3,
            "n_redundant": 0,
            "random_state": 13,
        },
        train_size=30,
        test_size=10,
        random_state=42,
        stratify=True,
        classifier=True,
        mode=LifelinesDataMode.AUXILIARY_MODEL,
        duration_col="T",
        event_col="E",
        benign_metric="accuracy",
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "survival_data.pkl"
        cfg.save_object(cfg, str(path))
        loaded = cfg.load_object(str(path))
    assert isinstance(loaded, LifelinesDataConfig)
    assert loaded.duration_col == "T"
    assert loaded.event_col == "E"
