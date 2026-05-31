import subprocess
import sys
from pathlib import Path

import pytest
from helpers import make_runtime_env


def _run_import_with_plugin_block(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    blocked_prefixes = (
        "deckard.plugins.anjana",
        "deckard.plugins.fairlearn",
        "deckard.plugins.lifelines",
        "deckard.plugins.openattack",
        "deckard.plugins.seaborn",
        "deckard.plugins.textattack",
        "deckard.plugins.yellowbrick",
    )
    script = f"""
import builtins

_real_import = builtins.__import__
_blocked_prefixes = {blocked_prefixes!r}


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name.startswith(prefix) for prefix in _blocked_prefixes):
        raise RuntimeError(f'blocked eager plugin import: {{name}}')
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _guarded_import
__import__({module_name!r})
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=make_runtime_env(repo_root / ".deckard_rc"),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"import {module_name} attempted eager plugin import\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "deckard.data",
        "deckard.model",
        "deckard.attack",
        "deckard.detector",
        "deckard.experiment",
        "deckard.score",
    ],
)
def test_core_packages_do_not_eagerly_import_plugin_families(module_name: str) -> None:
    _run_import_with_plugin_block(module_name)
