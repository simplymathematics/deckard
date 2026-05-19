import subprocess
import sys
from pathlib import Path

import pytest

from helpers import make_runtime_env


def _run_import_with_blocks(
    blocked_prefixes: tuple[str, ...], import_stmt: str
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = f"""
import builtins

_real_import = builtins.__import__
blocked = {blocked_prefixes!r}

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name.startswith(prefix) for prefix in blocked):
        raise ImportError(f\"blocked import: {{name}}\")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _guarded_import
{import_stmt}
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=make_runtime_env(repo_root / ".deckard_rc"),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"blocked prefixes={blocked_prefixes} import failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize(
    ("module_name", "symbol_name", "blocked_prefixes"),
    [
        (
            "deckard.plugins.anjana.score",
            "DefaultAnjanaScorerConfig",
            ("deckard.plugins.fairlearn", "deckard.plugins.lifelines"),
        ),
        (
            "deckard.plugins.fairlearn.score",
            "DefaultFairlearnScorerConfig",
            ("deckard.plugins.anjana", "deckard.plugins.lifelines"),
        ),
        (
            "deckard.plugins.lifelines.score",
            "DefaultLifelinesConfig",
            ("deckard.plugins.anjana", "deckard.plugins.fairlearn"),
        ),
    ],
)
def test_score_plugin_family_importable_without_sibling_families(
    module_name: str,
    symbol_name: str,
    blocked_prefixes: tuple[str, ...],
) -> None:
    _run_import_with_blocks(
        blocked_prefixes=blocked_prefixes,
        import_stmt=(
            f"module = __import__('{module_name}', fromlist=['{symbol_name}'])\n"
            f"assert getattr(module, '{symbol_name}') is not None"
        ),
    )
