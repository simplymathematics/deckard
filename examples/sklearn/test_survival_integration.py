import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lifelines") is None,
    reason="lifelines is required for survival integration tests",
)
@pytest.mark.parametrize(
    "dataset_name,survival_model",
    [
        ("leukemia", "weibull"),
        ("lung", "cox"),
        ("lifelines_diabetes", "weibull"),
        ("lifelines_diabetes", "cox"),
    ],
)
def test_survival_cli_in_examples_sklearn(dataset_name, survival_model, tmp_path):
    examples_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["DECKARD_CONFIG_DIR"] = "./config"
    env["DECKARD_DEFAULT_CONFIG_FILE"] = "survival.yaml"
    env["MPLBACKEND"] = "Agg"

    cmd = ["deckard", "survival", f"data={dataset_name}", f"model={survival_model}"]
    if shutil.which("deckard") is None:
        cmd = [
            sys.executable,
            "-m",
            "deckard",
            "survival",
            f"data={dataset_name}",
            f"model={survival_model}",
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
