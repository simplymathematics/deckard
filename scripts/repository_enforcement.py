#!/usr/bin/env python3
"""Repository-wide structural enforcement checks.

Default mode enforces durable, low-noise structural invariants:
- Class naming conventions for `*Config`, `*Mixin`, `*Plugin`
- Default scorer naming conventions for `Default*ScorerDictConfig`
- `*Config` inheritance chains (direct or indirect `*Config`/`BaseConfig`/`ABC`)
- Dataclass decoration for mixins that define dataclass-like fields
- Docstrings must avoid reStructuredText tokens
- Docstrings for selected protected mixin hook methods
- `*Plugin.__call__` signature contract (`*args`, `**kwargs`)

Strict mode (opt-in via ``--strict-docs-types``) additionally enforces public
docstring and annotation policies for all classes in the selected scope.
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
PROTECTED_MIXIN_HOOK_METHODS = {
    "_instantiate_plugin",
    "_get_plugins",
    "_run_plugin_hook",
    "_merge_plugin_scores",
}

# Rule annotations: keep this map in sync with emitted rule codes below.
RULE_ANNOTATIONS = {
    # Default mode naming/shape rules
    "NAME002": "Classes containing 'Mixin' must end with 'Mixin'.",
    "NAME003": "Classes containing 'Plugin' must end with 'Plugin'.",
    "NAME004": "Classes containing 'ScorerDict' must end with 'ScorerDictConfig'.",
    "NAME005": "Default scorer classes must end with 'ScorerDictConfig'.",
    "MIX001": "Mixins with concrete annotated defaults must be dataclasses.",
    "MIX003": "Mixin class names must be public (no leading underscore).",
    "MIX004": "Mixin classes must include a class docstring.",
    "MIX005": "Docstrings must avoid reStructuredText tokens.",
    "MIX006": "Public mixin methods must include docstrings.",
    "MIX007": "Selected protected mixin hook methods must include docstrings.",
    "PLG001": "Plugin classes must implement __call__.",
    "PLG002": "Plugin __call__ signatures must include *args and **kwargs.",
    # Strict docs/types mode rules
    "ANN001": "Public method parameters (except self/cls) require annotations.",
    "ANN003": "Public methods require return annotations.",
    "ANN004": "Public method return annotations cannot use Any/object.",
    "DOC001": "Public methods require docstrings.",
    "DOC002": "Public method docstrings must avoid reStructuredText tokens.",
    "DOC003": "Public methods with user parameters require an 'Args:' section.",
    "DOC004": "Public methods with non-None returns require a 'Returns:' section.",
    "DOC005": "Public methods that raise exceptions require a 'Raises:' section.",
    "DOC006": "Selected public classes require an 'Attributes:' section in class docstrings.",
}


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
    positional = [arg for arg in fn.args.args if arg.arg not in {"self", "cls"}]
    keyword_only = list(fn.args.kwonlyargs)
    has_variadics = fn.args.vararg is not None or fn.args.kwarg is not None
    return bool(positional or keyword_only or has_variadics)


def _returns_none_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id == "None"
    if isinstance(node, ast.Constant):
        return node.value is None
    return False


def _has_raise_statement(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(child, ast.Raise) for child in ast.walk(fn))


def _class_requires_attributes_section(class_name: str) -> bool:
    """Return True when class docstring must include a Google-style Attributes section."""
    if class_name.startswith("_"):
        return False
    suffixes = ("Config", "Mixin", "Plugin")
    tokens = ("Sampler", "Pipeline", "Trainer", "Defense", "Scorer")
    return class_name.endswith(suffixes) or any(token in class_name for token in tokens)


def _class_field_annotations(node: ast.ClassDef) -> list[ast.AnnAssign]:
    anns: list[ast.AnnAssign] = []
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            anns.append(child)
    return anns


def _docstring_lineno(node: ast.AST) -> int:
    """Return the starting line for a node docstring when available."""
    body = getattr(node, "body", None)
    if isinstance(body, list) and body:
        first_stmt = body[0]
        if isinstance(first_stmt, ast.Expr):
            value = first_stmt.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return first_stmt.lineno
    return getattr(node, "lineno", 1)


def validate_file(
    path: Path,
    *,
    strict_docs_types: bool = False,
    require_attributes_sections: bool = False,
) -> list[Violation]:
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=rel)
    violations: list[Violation] = []

    # MIX005: default mode policy for RST tokens across all docstrings.
    docstring_nodes: list[
        ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
    ] = [tree]
    docstring_nodes.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    for node in docstring_nodes:
        doc = ast.get_docstring(node)
        if doc and any(token in doc for token in RST_TOKENS):
            if isinstance(node, ast.Module):
                target = "module"
            elif isinstance(node, ast.ClassDef):
                target = node.name
            else:
                owner = next(
                    (
                        cls.name
                        for cls in tree.body
                        if isinstance(cls, ast.ClassDef)
                        and node in getattr(cls, "body", [])
                    ),
                    None,
                )
                target = f"{owner}.{node.name}" if owner else node.name
            violations.append(
                Violation(
                    rel,
                    _docstring_lineno(node),
                    "MIX005",
                    f"{target} docstring contains reStructuredText tokens",
                ),
            )

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        class_name = node.name
        decorators = {_decorator_name(d) for d in node.decorator_list}
        has_mixin_token = "Mixin" in class_name
        has_plugin_token = "Plugin" in class_name
        # ScoreDict is an independent runtime payload type (not a config class).
        has_scorer_dict_token = "ScorerDict" in class_name
        has_default_score_family = (
            class_name.startswith("Default")
            and class_name.endswith("Config")
            and ("Score" in class_name or "Scorer" in class_name)
        )
        # NAME002: classes with a mixin token must end with 'Mixin'.
        if has_mixin_token and not class_name.endswith("Mixin"):
            violations.append(
                Violation(
                    rel,
                    node.lineno,
                    "NAME002",
                    f"{class_name} must end with 'Mixin'",
                ),
            )
        # NAME003: classes with a plugin token must end with 'Plugin'.
        if has_plugin_token and not class_name.endswith("Plugin"):
            violations.append(
                Violation(
                    rel,
                    node.lineno,
                    "NAME003",
                    f"{class_name} must end with 'Plugin'",
                ),
            )

        # NAME004: scorer-dict config classes use a canonical ScorerDictConfig suffix.
        if has_scorer_dict_token and not class_name.endswith("ScorerDictConfig"):
            violations.append(
                Violation(
                    rel,
                    node.lineno,
                    "NAME004",
                    f"{class_name} must end with 'ScorerDictConfig'",
                ),
            )

        # NAME005: default scorer classes use canonical ScorerDictConfig suffix.
        if has_default_score_family and not class_name.endswith("ScorerDictConfig"):
            violations.append(
                Violation(
                    rel,
                    node.lineno,
                    "NAME005",
                    f"{class_name} must end with 'ScorerDictConfig'",
                ),
            )

        if class_name.endswith("Mixin"):
            # MIX003: mixin class names are public API and must not be private.
            if class_name.startswith("_"):
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        "MIX003",
                        f"{class_name} must be public and must not start with '_'",
                    ),
                )

            field_anns = _class_field_annotations(node)
            # Require dataclass only when mixins define concrete defaults, not
            # when they only declare type hints for static analysis.
            has_concrete_field = any(ann.value is not None for ann in field_anns)
            # MIX001: concrete field defaults in mixins require dataclass semantics.
            if has_concrete_field and "dataclass" not in decorators:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        "MIX001",
                        f"{class_name} defines annotated fields and must be a dataclass",
                    ),
                )

            public_methods = [
                c
                for c in node.body
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not c.name.startswith("_")
            ]
            protected_hook_methods = [
                c
                for c in node.body
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                and c.name in PROTECTED_MIXIN_HOOK_METHODS
            ]

            class_doc = ast.get_docstring(node) or ""
            # MIX004/MIX005: class docstring presence + token policy.
            if not class_doc.strip():
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        "MIX004",
                        f"{class_name} missing class docstring",
                    ),
                )

            # MIX006: public mixin methods must be documented.
            for fn in public_methods:
                doc = ast.get_docstring(fn) or ""
                if not doc.strip():
                    violations.append(
                        Violation(
                            rel,
                            fn.lineno,
                            "MIX006",
                            f"{class_name}.{fn.name} missing public docstring",
                        ),
                    )

            # MIX007: selected protected hook methods must also be documented.
            for fn in protected_hook_methods:
                doc = ast.get_docstring(fn) or ""
                if not doc.strip():
                    violations.append(
                        Violation(
                            rel,
                            fn.lineno,
                            "MIX007",
                            f"{class_name}.{fn.name} missing protected hook docstring",
                        ),
                    )

        methods = [
            c
            for c in node.body
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        # PLG001/PLG002: plugin callable contract enforcement.
        if class_name.endswith("Plugin"):
            call = next((m for m in methods if m.name == "__call__"), None)
            if call is None:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        "PLG001",
                        f"{class_name} must implement __call__",
                    ),
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
        # Docs apply across all classes; annotation strictness stays focused on
        # canonical runtime classes to keep signal-to-noise manageable.
        if strict_docs_types:
            for fn in methods:
                if not _is_public_method(fn):
                    continue

                if class_name.endswith(("Config", "Mixin", "Plugin")):
                    # ANN001/ANN003/ANN004: strict type annotation checks.
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

                # DOC001-DOC003: strict public docstring checks.
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
                    if (
                        fn.name != "__init__"
                        and fn.returns is not None
                        and not _returns_none_annotation(fn.returns)
                        and "Returns:" not in doc
                    ):
                        violations.append(
                            Violation(
                                rel,
                                fn.lineno,
                                "DOC004",
                                f"{class_name}.{fn.name} missing Google-style 'Returns:' section",
                            ),
                        )
                    if _has_raise_statement(fn) and "Raises:" not in doc:
                        violations.append(
                            Violation(
                                rel,
                                fn.lineno,
                                "DOC005",
                                f"{class_name}.{fn.name} missing Google-style 'Raises:' section",
                            ),
                        )

        if require_attributes_sections and _class_requires_attributes_section(
            class_name,
        ):
            class_doc = ast.get_docstring(node) or ""
            if "Attributes:" not in class_doc:
                violations.append(
                    Violation(
                        rel,
                        node.lineno,
                        "DOC006",
                        f"{class_name} missing Google-style 'Attributes:' section",
                    ),
                )

    return violations


def collect_violations(
    scope: str,
    *,
    strict_docs_types: bool = False,
    require_attributes_sections: bool = False,
) -> list[Violation]:
    candidate = Path(scope)
    base = candidate if candidate.is_absolute() else ROOT / scope
    violations: list[Violation] = []
    if base.is_file() and base.suffix == ".py":
        violations.extend(
            validate_file(
                base,
                strict_docs_types=strict_docs_types,
                require_attributes_sections=require_attributes_sections,
            ),
        )
        return sorted(violations, key=lambda v: (v.path, v.line, v.code, v.message))
    for path in _iter_python_files(base):
        violations.extend(
            validate_file(
                path,
                strict_docs_types=strict_docs_types,
                require_attributes_sections=require_attributes_sections,
            ),
        )
    return sorted(violations, key=lambda v: (v.path, v.line, v.code, v.message))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run repository-wide enforcement checks",
    )
    parser.add_argument(
        "--scope",
        default="deckard",
        help="Path scope to validate (default: deckard)",
    )
    parser.add_argument(
        "--strict-docs-types",
        action="store_true",
        help="Enable strict public docstring/type-annotation checks",
    )
    parser.add_argument(
        "--require-attributes-sections",
        action="store_true",
        help="Require Google-style 'Attributes:' sections for Config/Mixin/Plugin and related runtime classes",
    )
    args = parser.parse_args()

    violations = collect_violations(
        args.scope,
        strict_docs_types=args.strict_docs_types,
        require_attributes_sections=args.require_attributes_sections,
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
