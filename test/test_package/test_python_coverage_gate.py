from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "python_coverage_gate.py"


def _write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_coverage_gate_passes_for_data_threshold(tmp_path: Path) -> None:
    report = tmp_path / "coverage.txt"
    _write_report(
        report,
        [
            "Name Stmts Miss Cover Missing",
            "-----------------------------",
            "deckard/data/base.py 100 10 90% 1-2",
            "deckard/data/sample.py 50 10 80% 4-5",
            "TOTAL 150 20 87%",
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--report",
            str(report),
            "--threshold",
            "80",
            "--prefix",
            "deckard/data",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Coverage gate passed." in result.stdout


def test_coverage_gate_fails_for_framework_plugin_groups(tmp_path: Path) -> None:
    report = tmp_path / "coverage.txt"
    _write_report(
        report,
        [
            "Name Stmts Miss Cover Missing",
            "-----------------------------",
            "deckard/data/base.py 100 20 80% 1-2",
            "deckard/frameworks/pytorch/model.py 100 40 60% 8-20",
            "deckard/frameworks/transformers/model.py 100 70 30% 8-20",
            "deckard/plugins/fairlearn/score.py 100 40 60% 8-20",
            "deckard/plugins/yellowbrick/plot.py 100 50 50% 8-20",
            "TOTAL 500 220 56%",
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--report",
            str(report),
            "--threshold",
            "80",
            "--prefix",
            "deckard/data",
            "--enforce-frameworks",
            "--enforce-plugins",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "GROUP deckard/frameworks/pytorch" in result.stdout
    assert "GROUP deckard/plugins/fairlearn" in result.stdout
