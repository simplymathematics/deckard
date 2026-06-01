from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_fixer(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            "scripts/fix_dataclass_field_rules.py",
            "--scope",
            str(path),
            *args,
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_fix_dataclass_field_rules_adds_init_false_and_metadata(tmp_path: Path) -> None:
    sample = tmp_path / "runtime_fields.py"
    sample.write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        "    _model: object | None = None\n"
        "    user_value: int = 1\n",
        encoding="utf-8",
    )

    result = _run_fixer(sample)

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    text = sample.read_text(encoding="utf-8")
    assert "from dataclasses import dataclass, field" in text
    assert "_model: object | None = field(" in text
    assert "default=None" in text
    assert "init=False" in text
    assert "document _model" in text


def test_fix_dataclass_field_rules_preserves_existing_metadata(tmp_path: Path) -> None:
    sample = tmp_path / "metadata_fields.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        "    value: int = field(default=1, metadata={\"help\": \"Keep me\"})\n",
        encoding="utf-8",
    )

    result = _run_fixer(sample, "--fix-cfg008")

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    text = sample.read_text(encoding="utf-8")
    assert 'metadata={"help": "Keep me"}' in text


def test_fix_dataclass_field_rules_dry_run_does_not_write(tmp_path: Path) -> None:
    sample = tmp_path / "dry_run_fields.py"
    original = (
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        "    _model: object | None = None\n"
    )
    sample.write_text(original, encoding="utf-8")

    result = _run_fixer(sample, "--dry-run")

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert sample.read_text(encoding="utf-8") == original


def test_fix_dataclass_field_rules_preserves_module_docstring_position(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "docstring_fields.py"
    sample.write_text(
        '"""Module docstring."""\n\n'
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        "    _model: object | None = None\n",
        encoding="utf-8",
    )

    result = _run_fixer(sample)

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    text = sample.read_text(encoding="utf-8")
    assert text.startswith('"""Module docstring."""')
    assert 'from dataclasses import dataclass, field' in text


def test_fix_dataclass_field_rules_adds_repr_false_for_cfg009(tmp_path: Path) -> None:
    sample = tmp_path / "runtime_repr_fields.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        "    runtime_state: int = field(default=1, init=False)\n",
        encoding="utf-8",
    )

    result = _run_fixer(sample, "--fix-cfg009")

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    text = sample.read_text(encoding="utf-8")
    assert "init=False" in text
    assert "repr=False" in text


def test_fix_dataclass_field_rules_cfg009_skips_target_field(tmp_path: Path) -> None:
    sample = tmp_path / "target_field.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        "    _target_: str | None = field(default=\"target_field.RuntimeConfig\", init=False)\n",
        encoding="utf-8",
    )

    result = _run_fixer(sample, "--fix-cfg009")

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    text = sample.read_text(encoding="utf-8")
    assert "repr=False" not in text
