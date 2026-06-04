from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "deckard"


def _iter_python_files(relative_glob: str):
    yield from ROOT.glob(relative_glob)


def _imported_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
    return modules


def test_canon_modules_do_not_import_base_modules() -> None:
    canon_files = list(_iter_python_files("*/canon.py"))
    violations: list[str] = []
    for path in canon_files:
        modules = _imported_modules(path)
        if any(module.endswith(".base") for module in modules):
            violations.append(path.relative_to(ROOT).as_posix())

    assert (
        violations == []
    ), "canon modules must not import base modules; violations: " + ", ".join(
        sorted(violations),
    )


def test_utils_and_orchestration_do_not_import_domain_base_modules() -> None:
    for module_file in (ROOT / "utils.py", ROOT / "orchestration.py"):
        modules = _imported_modules(module_file)
        violations = sorted(module for module in modules if module.endswith(".base"))
        assert violations == [], (
            f"{module_file.name} must not import domain base modules; violations: "
            + ", ".join(violations)
        )
