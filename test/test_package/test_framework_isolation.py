import subprocess
import sys
from pathlib import Path

from helpers import make_runtime_env


def _run_import_with_block(blocked_prefix: str, import_stmt: str):
    repo_root = Path(__file__).resolve().parents[2]
    script = f"""
import builtins

_real_import = builtins.__import__

def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.startswith({blocked_prefix!r}):
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
        f"blocked prefix={blocked_prefix} import failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_sklearn_framework_importable_without_pytorch_module():
    _run_import_with_block(
        blocked_prefix="deckard.frameworks.pytorch",
        import_stmt="from deckard.frameworks.sklearn import DefaultSklearnDefenseConfig\nassert DefaultSklearnDefenseConfig is not None",
    )


def test_pytorch_framework_importable_without_sklearn_module():
    _run_import_with_block(
        blocked_prefix="deckard.frameworks.sklearn",
        import_stmt="from deckard.frameworks.pytorch import DefaultPytorchDefenseConfig\nassert DefaultPytorchDefenseConfig is not None",
    )
