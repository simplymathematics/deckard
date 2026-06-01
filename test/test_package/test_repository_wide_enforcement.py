"""Repository-wide enforcement tests.

These tests validate structural and runtime enforcement guarantees introduced
for plugin/framework decoupling and deterministic orchestration.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from deckard.declarations import (
    register_configs,
)
from deckard.plugins import HookPlugin
from deckard.plugins.anjana.data import AnjanaDataConfig
from deckard.plugins.fairlearn.data import FairlearnDataConfig


RELAXED_ENFORCEMENT_FLAGS = (
    "--no-enforce-class-contracts",
    "--no-strict-docs-types",
    "--no-require-attributes-sections",
    "--no-enforce-runtime-init-params",
    "--no-enforce-field-metadata",
    "--no-enforce-runtime-repr",
    "--no-enforce-target-field",
)


def _run_enforcement(scope: str, *extra_args: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "scripts/repository_enforcement.py",
        "--scope",
        scope,
        "--docs-scope",
        "none",
        *extra_args,
    ]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def _run_enforcement_result(
    scope: str,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "scripts/repository_enforcement.py",
        "--scope",
        scope,
        "--docs-scope",
        "none",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_repository_enforcement_plugins_scope_passes() -> None:
    _run_enforcement("deckard/plugins", *RELAXED_ENFORCEMENT_FLAGS)


def test_repository_enforcement_frameworks_scope_passes() -> None:
    _run_enforcement("deckard/frameworks", *RELAXED_ENFORCEMENT_FLAGS)


def test_repository_enforcement_score_scope_passes() -> None:
    _run_enforcement("deckard/score", *RELAXED_ENFORCEMENT_FLAGS)


def test_repository_enforcement_deckard_scope_passes() -> None:
    _run_enforcement("deckard", *RELAXED_ENFORCEMENT_FLAGS)


def test_repository_enforcement_default_score_config_name_fails(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "bad_default_score_config.py"
    sample.write_text(
        "class DefaultModelScoreConfig:\n"
        '    """Temporary test class."""\n'
        "    pass\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(
        str(sample),
        "--enforce-class-contracts",
    )

    assert result.returncode == 1
    assert "NAME005" in result.stdout
    assert "must end with 'ScorerDictConfig'" in result.stdout


def test_repository_enforcement_default_scorer_dict_config_name_passes(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "good_default_scorer_dict_config.py"
    sample.write_text(
        "class DefaultModelScorerDictConfig:\n"
        '    """Temporary test class.\n\n'
        "    Attributes:\n"
        "        score_name: Example score identifier.\n"
        '    """\n'
        "    score_name: str\n"
        "    pass\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(
        str(sample),
        "--enforce-class-contracts",
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_repository_enforcement_attributes_section_required_for_config(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "bad_attributes_config.py"
    sample.write_text(
        "class RuntimeConfig:\n"
        '    """Temporary config class without attributes section."""\n'
        "    value: int = 1\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(
        str(sample),
        "--enforce-class-contracts",
    )

    assert result.returncode == 1
    assert "DOC006" in result.stdout
    assert "missing Google-style 'Attributes:' section" in result.stdout


def test_repository_enforcement_attributes_section_passes_for_sampler(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "good_attributes_sampler.py"
    sample.write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class KFoldSampler:\n"
        '    """Temporary sampler with attributes section.\n\n'
        "    Attributes:\n"
        "        n_splits: Number of folds.\n"
        '    """\n'
        "    n_splits: int = 5\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(
        str(sample),
        "--enforce-class-contracts",
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_repository_enforcement_public_class_docstring_required(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "bad_public_class_docstring.py"
    sample.write_text(
        "class RuntimePolicy:\n" "    value: int = 1\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(
        str(sample),
        "--enforce-class-contracts",
    )

    assert result.returncode == 1
    assert "CLS004" in result.stdout
    assert "RuntimePolicy missing class docstring" in result.stdout


def test_repository_enforcement_public_class_docstring_passes(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "good_public_class_docstring.py"
    sample.write_text(
        "class RuntimePolicy:\n"
        '    """Temporary runtime policy."""\n'
        "    value: int\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(
        str(sample),
        "--enforce-class-contracts",
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_repository_enforcement_public_method_docstring_required(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "bad_public_method_docstring.py"
    sample.write_text(
        "class RuntimePolicy:\n"
        '    """Temporary runtime policy."""\n\n'
        "    def run(self) -> None:\n"
        "        pass\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(
        str(sample),
        "--enforce-class-contracts",
    )

    assert result.returncode == 1
    assert "CLS006" in result.stdout
    assert "RuntimePolicy.run missing public docstring" in result.stdout


def test_repository_enforcement_runtime_fields_must_be_init_false(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "bad_runtime_fields.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        '    """Temporary runtime config."""\n'
        "    _model: object | None = None\n"
        "    user_param: int = 1\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(str(sample))

    assert result.returncode == 1
    assert "CFG007" in result.stdout
    assert "RuntimeConfig._model" in result.stdout


def test_repository_enforcement_runtime_fields_init_false_passes(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "good_runtime_fields.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        '    """Temporary runtime config.\n\n'
        "    Attributes:\n"
        "        _model: Runtime model object.\n"
        "        user_param: Example user parameter.\n"
        '    """\n'
        '    _model: object | None = field(default=None, init=False, repr=False, metadata={"help": "Runtime model"})\n'
        "    user_param: int = 1\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(str(sample))

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_repository_enforcement_field_metadata_required(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "bad_field_metadata.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        '    """Temporary runtime config."""\n'
        "    value: int = field(default=1)\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(str(sample))

    assert result.returncode == 1
    assert "CFG008" in result.stdout
    assert "RuntimeConfig.value" in result.stdout


def test_repository_enforcement_field_metadata_passes(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "good_field_metadata.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        '    """Temporary runtime config.\n\n'
        "    Attributes:\n"
        "        value: Example runtime value.\n"
        '    """\n'
        '    value: int = field(default=1, metadata={"help": "Example"})\n',
        encoding="utf-8",
    )

    result = _run_enforcement_result(str(sample))

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_repository_enforcement_runtime_repr_required(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "bad_runtime_repr.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        '    """Temporary runtime config."""\n'
        "    runtime_state: int = field(default=1, init=False)\n",
        encoding="utf-8",
    )

    result = _run_enforcement_result(str(sample))

    assert result.returncode == 1
    assert "CFG009" in result.stdout
    assert "RuntimeConfig.runtime_state" in result.stdout


def test_repository_enforcement_runtime_repr_passes(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "good_runtime_repr.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        '    """Temporary runtime config.\n\n'
        "    Attributes:\n"
        "        runtime_state: Example runtime state.\n"
        '    """\n'
        '    runtime_state: int = field(default=1, init=False, repr=False, metadata={"help": "Runtime state"})\n',
        encoding="utf-8",
    )

    result = _run_enforcement_result(str(sample))

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_repository_enforcement_target_field_required(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "bad_target_field.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        '    """Temporary runtime config."""\n'
        '    _target_: str | None = field(default="deckard.alias.RuntimeConfig", init=False, repr=False, metadata={"help": "Example"})\n',
        encoding="utf-8",
    )

    result = _run_enforcement_result(str(sample))

    assert result.returncode == 1
    assert "CFG010" in result.stdout
    assert "bad_target_field.RuntimeConfig" in result.stdout


def test_repository_enforcement_target_field_passes(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "good_target_field.py"
    sample.write_text(
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class RuntimeConfig:\n"
        '    """Temporary runtime config.\n\n'
        "    Attributes:\n"
        "        _target_: Canonical runtime path.\n"
        '    """\n'
        '    _target_: str | None = field(default="good_target_field.RuntimeConfig", metadata={"help": "Example"})\n',
        encoding="utf-8",
    )

    result = _run_enforcement_result(str(sample))

    assert result.returncode == 0, result.stdout + "\n" + result.stderr


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

    assert anjana_mro.index("PrivacyBehaviorMixin") < anjana_mro.index(
        "DataConfig",
    )
    assert fairlearn_mro.index("FairnessBehaviorMixin") < fairlearn_mro.index(
        "DataConfig",
    )


def test_runtime_declaration_registration_is_idempotent() -> None:
    # register_configs intentionally logs-and-skips duplicate group/name pairs.
    # Running it repeatedly should not raise and should remain deterministic.
    register_configs()
    register_configs()
