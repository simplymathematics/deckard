from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_crosslink_fixer(path: Path) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            "scripts/fix_docs_crosslinks.py",
            "--scope",
            str(path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_crosslink_fixer_preserves_kwarg_like_inline_code(tmp_path: Path) -> None:
    sample = tmp_path / "example.md"
    original = "Set `x` and `hue` kwargs before calling `train`.\n"
    sample.write_text(original, encoding="utf-8")

    result = _run_crosslink_fixer(sample)

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert sample.read_text(encoding="utf-8") == original


def test_crosslink_fixer_still_links_real_symbols(tmp_path: Path) -> None:
    sample = tmp_path / "example.md"
    sample.write_text("Use `ModelConfig` for the model wrapper.\n", encoding="utf-8")

    result = _run_crosslink_fixer(sample)

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    text = sample.read_text(encoding="utf-8")
    assert "[ModelConfig](" in text
    assert "TODO-BROKEN-LINK" not in text
    assert "docs/api/model/index" in text


def test_crosslink_fixer_falls_back_to_api_modules_index_for_ambiguous_symbols(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "example.md"
    sample.write_text("Use `save` to persist results.\n", encoding="utf-8")

    result = _run_crosslink_fixer(sample)

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    text = sample.read_text(encoding="utf-8")
    assert "[save](" in text
    assert "TODO-BROKEN-LINK" not in text
    assert "docs/api/modules" in text
