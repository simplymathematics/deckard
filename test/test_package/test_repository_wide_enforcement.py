"""Repository-wide enforcement tests.

These tests validate structural and runtime enforcement guarantees introduced
for plugin/framework decoupling and deterministic orchestration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from deckard.declarations import (
    register_configs,
)
from deckard.plugins import HookPlugin
from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.fairlearn.data import FairlearnDataConfig


def _run_enforcement(scope: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        "python",
        "scripts/repository_enforcement.py",
        "--scope",
        scope,
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_repository_enforcement_plugins_scope_passes() -> None:
    _run_enforcement("deckard/plugins")


def test_repository_enforcement_frameworks_scope_passes() -> None:
    _run_enforcement("deckard/frameworks")


def test_plugin_orchestration_is_deterministic() -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.events: list[str] = []

        def first(self) -> str:
            self.events.append("first")
            return "first"

        def second(self) -> str:
            self.events.append("second")
            return "second"

    runtime = _Runtime()
    plugins = [
        HookPlugin(hook_name="first", method_name="first"),
        HookPlugin(hook_name="second", method_name="second"),
    ]

    seq1 = [plugin(runtime, hook_name=plugin.hook_name) for plugin in plugins]
    seq2 = [plugin(runtime, hook_name=plugin.hook_name) for plugin in plugins]

    assert seq1 == ["first", "second"]
    assert seq2 == ["first", "second"]
    assert runtime.events == ["first", "second", "first", "second"]


def test_mixin_composition_order_is_explicit() -> None:
    anjana_mro = [cls.__name__ for cls in AnjanaDataConfig.__mro__]
    fairlearn_mro = [cls.__name__ for cls in FairlearnDataConfig.__mro__]

    assert anjana_mro.index("_PrivacyBehaviorMixin") < anjana_mro.index(
        "DataPipelineConfig",
    )
    assert fairlearn_mro.index("_FairnessBehaviorMixin") < fairlearn_mro.index(
        "DataPipelineConfig",
    )


def test_runtime_declaration_registration_is_idempotent() -> None:
    # register_configs intentionally logs-and-skips duplicate group/name pairs.
    # Running it repeatedly should not raise and should remain deterministic.
    register_configs()
    register_configs()
