#!/usr/bin/env python3
"""Repository-wide structural enforcement checks.

Default mode enforces durable, low-noise structural invariants:
- Class naming conventions for `*Config`, `*Mixin`, `*Plugin`
- `*Config` inheritance chains (direct or indirect `*Config`/`ConfigBase`/`ABC`)
- Dataclass decoration for mixins that define dataclass-like fields
- `*Plugin.__call__` signature contract (`*args`, `**kwargs`)

Strict mode (opt-in via ``--strict-docs-types``) additionally enforces public
docstring and annotation policies.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DECKARD_DIR = ROOT / "deckard"

RST_TOKENS = (":param", ":type", ":rtype:", ".. code-block::", ".. note::")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    code: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"


def _iter_python_files(base: Path) -> Iterable[Path]:
    for path in base.rglob("*.py"):
        rel = path.relative_to(ROOT)
        if "build/" in rel.as_posix():
            continue
        if "__pycache__" in rel.as_posix():
            continue
        yield path


def _decorator_name(dec: ast.expr) -> str:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    return ""


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Subscript):
        return _base_name(base.value)
    return ""


def _annotation_contains_forbidden(node: ast.AST | None) -> bool:
    if node is None:
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in {"Any", "object"}:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in {"Any"}:
            return True
    return False


def _has_vararg_and_kwarg(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return fn.args.vararg is not None and fn.args.kwarg is not None


def _is_public_method(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return not fn.name.startswith("_") or fn.name == "__call__"


def _has_user_parameters(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    positional = [
        arg
        for arg in fn.args.args
        if arg.arg not in {"self", "cls"}
    ]
    keyword_only = list(fn.args.kwonlyargs)
    has_variadics = fn.args.vararg is not None or fn.args.kwarg is not None
    return bool(positional or keyword_only or has_variadics)


def _is_variadic_arg_name(arg_name: str) -> bool:
    return arg_name in {"args", "kwargs"}


def _class_field_annotations(node: ast.ClassDef) -> list[ast.AnnAssign]:
    anns: list[ast.AnnAssign] = []
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            anns.append(child)
    return anns


def validate_file(path: Path, *, strict_docs_types: bool = False) -> list[Violation]:
    rel = path.relative_to(ROOT).as_posix()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=rel)
    violations: list[Violation] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        class_name = node.name
        base_names = [_base_name(b) for b in node.bases]
        decorators = {_decorator_name(d) for d in node.decorator_list}

        has_config_token = "Config" in class_name
        has_mixin_token = "Mixin" in class_name
        has_plugin_token = "Plugin" in class_name

        if has_config_token and not has_mixin_token and not (
            class_name.endswith("Config")
            or class_name.endswith("ConfigList")
            or class_name.endswith("Contract")
        ):
            violations.append(
                Violation(rel, node.lineno, "NAME001", f"{class_name} must end with 'Config'"),
            )
        if has_mixin_token and not class_name.endswith("Mixin"):
            violations.append(
                Violation(rel, node.lineno, "NAME002", f"{class_name} must end with 'Mixin'"),
            )
        if has_plugin_token and not class_name.endswith("Plugin"):
            violations.append(
                Violation(rel, node.lineno, "NAME003", f"{class_name} must end with 'Plugin'"),
            )

        if class_name.endswith("Config"):
            inherits_ok = any(
                base.endswith("Config") or base == "ConfigBase" or base == "ABC"
                for base in base_names
            )
            if not inherits_ok:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        "CFG001",
                        f"{class_name} should inherit from ConfigBase or another *Config base",
                    ),
                )

        if class_name.endswith("Mixin"):
            field_anns = _class_field_annotations(node)
            # Require dataclass only when mixins define concrete defaults, not
            # when they only declare type hints for static analysis.
            has_concrete_field = any(ann.value is not None for ann in field_anns)
            if has_concrete_field and "dataclass" not in decorators:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        "MIX001",
                        f"{class_name} defines annotated fields and must be a dataclass",
                    ),
                )

            # Public *Mixin classes (no leading underscore) must expose at
            # least one non-dunder, non-underscore-prefixed method so that
            # adapter callers have a stable public API surface.
            if not class_name.startswith("_"):
                public_methods = [
                    c
                    for c in node.body
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not c.name.startswith("_")
                ]
                if not public_methods:
                    violations.append(
                        Violation(
                            rel,
                            node.lineno,
                            "MIX002",
                            f"{class_name} must expose at least one public-facing method",
                        ),
                    )

        methods = [
            c
            for c in node.body
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if class_name.endswith("Plugin"):
            call = next((m for m in methods if m.name == "__call__"), None)
            if call is None:
                violations.append(
                    Violation(rel, node.lineno, "PLG001", f"{class_name} must implement __call__"),
                )
            else:
                if not _has_vararg_and_kwarg(call):
                    violations.append(
                        Violation(
                            rel,
                            call.lineno,
                            "PLG002",
                            f"{class_name}.__call__ must include *args and **kwargs",
                        ),
                    )

        # Optional strict checks for docs and annotations.
        if strict_docs_types and class_name.endswith(("Config", "Mixin", "Plugin")):
            for fn in methods:
                if not _is_public_method(fn):
                    continue

                # Type annotations on args + returns.
                all_args = [*fn.args.args, *fn.args.kwonlyargs]
                for arg in all_args:
                    if arg.arg in {"self", "cls"}:
                        continue
                    if arg.annotation is None:
                        violations.append(
                            Violation(
                                rel,
                                fn.lineno,
                                "ANN001",
                                f"{class_name}.{fn.name} missing annotation for '{arg.arg}'",
                            ),
                        )
                    elif _annotation_contains_forbidden(arg.annotation):
                        if _is_variadic_arg_name(arg.arg):
                            # Variadic runtime payloads are intentionally
                            # permissive in this repository: `*args`/`**kwargs`
                            # may use `Any` by contract.
                            continue
                        violations.append(
                            Violation(
                                rel,
                                fn.lineno,
                                "ANN002",
                                f"{class_name}.{fn.name} uses forbidden type annotation Any/object",
                            ),
                        )
                if fn.returns is None:
                    violations.append(
                        Violation(
                            rel,
                            fn.lineno,
                            "ANN003",
                            f"{class_name}.{fn.name} missing return annotation",
                        ),
                    )
                elif _annotation_contains_forbidden(fn.returns):
                    violations.append(
                        Violation(
                            rel,
                            fn.lineno,
                            "ANN004",
                            f"{class_name}.{fn.name} uses forbidden return annotation Any/object",
                        ),
                    )

                doc = ast.get_docstring(fn) or ""
                if not doc.strip():
                    violations.append(
                        Violation(
                            rel,
                            fn.lineno,
                            "DOC001",
                            f"{class_name}.{fn.name} missing public docstring",
                        ),
                    )
                else:
                    if any(token in doc for token in RST_TOKENS):
                        violations.append(
                            Violation(
                                rel,
                                fn.lineno,
                                "DOC002",
                                f"{class_name}.{fn.name} contains reStructuredText tokens",
                            ),
                        )
                    # Require an Args section when public methods expose
                    # user-facing parameters beyond self/cls.
                    if (
                        fn.name != "__init__"
                        and _has_user_parameters(fn)
                        and "Args:" not in doc
                    ):
                        violations.append(
                            Violation(
                                rel,
                                fn.lineno,
                                "DOC003",
                                f"{class_name}.{fn.name} missing Google-style 'Args:' section",
                            ),
                        )

    return violations


def collect_violations(scope: str, *, strict_docs_types: bool = False) -> list[Violation]:
    base = ROOT / scope
    violations: list[Violation] = []
    for path in _iter_python_files(base):
        violations.extend(validate_file(path, strict_docs_types=strict_docs_types))
    return sorted(violations, key=lambda v: (v.path, v.line, v.code, v.message))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository-wide enforcement checks")
    parser.add_argument(
        "--scope",
        default="deckard/plugins",
        help="Path scope to validate (default: deckard/plugins)",
    )
    parser.add_argument(
        "--strict-docs-types",
        action="store_true",
        help="Enable strict public docstring/type-annotation checks",
    )
    args = parser.parse_args()

    violations = collect_violations(
        args.scope,
        strict_docs_types=args.strict_docs_types,
    )
    if violations:
        for violation in violations:
            print(violation.format())
        print(f"\nFound {len(violations)} enforcement violations in {args.scope}")
        return 1

    print(f"No enforcement violations found in {args.scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
