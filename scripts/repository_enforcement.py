#!/usr/bin/env python3
"""Repository-wide structural enforcement checks.

Default mode enforces all repository structural, docs, and runtime invariants:
- Class naming conventions for `*Config`, `*Mixin`, `*Plugin`
- Default scorer naming conventions for `Default*ScorerDictConfig`
- `*Config` inheritance chains (direct or indirect `*Config`/`BaseConfig`/`ABC`)
- Dataclass decoration for mixins that define dataclass-like fields
- Runtime dataclass fields are distinct from initialization params
- Docstrings must avoid reStructuredText tokens
- Docstrings for selected protected mixin hook methods
- `*Plugin.__call__` signature contract (`*args`, `**kwargs`)
- Public class/method docstring and annotation policies
- Dataclass field metadata and runtime repr conventions
- Canonical `_target_` field semantics
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DECKARD_DIR = ROOT / "deckard"
DEFAULT_DOCS_SCOPES = ("docs",)

RST_TOKENS = (":param", ":type", ":rtype:", ".. code-block::", ".. note::")
PROTECTED_MIXIN_HOOK_METHODS = {
    "_instantiate_plugin",
    "_get_plugins",
    "_run_plugin_hook",
    "_merge_plugin_scores",
}

RUNTIME_FIELD_NAMES = {
    "score_dict",
    "_model",
    "_model_config",
    "_plugin_objects",
    "_runtime_defense_state",
    "_defense_applied_at",
}

# Rule annotations: keep this map in sync with emitted rule codes below.
RULE_ANNOTATIONS = {
    # Default mode naming/shape rules
    "NAME002": "Classes containing 'Mixin' must end with 'Mixin'.",
    "NAME003": "Classes containing 'Plugin' must end with 'Plugin'.",
    "NAME004": "Classes containing 'ScorerDict' must end with 'ScorerDictConfig'.",
    "NAME005": "Default scorer classes must end with 'ScorerDictConfig'.",
    "CLS001": "Public classes with concrete annotated defaults must be dataclasses.",
    "CLS004": "Public classes must include a class docstring.",
    "CLS005": "Docstrings must avoid reStructuredText tokens.",
    "CLS006": "Public class methods must include docstrings.",
    "CLS007": "Selected protected hook methods must include docstrings.",
    "CFG007": "Runtime dataclass fields must use init=False.",
    "CFG008": "Dataclass field() declarations must include metadata.",
    "CFG009": "Dataclass field(init=False) declarations except _target_ must also set repr=False.",
    "CFG010": "Dataclass _target_ fields must preserve init/repr visibility and default to the canonical object path.",
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
    # Docs mode rules
    "DOCX001": "Use MyST role syntax {mod}`...` instead of :mod:`...` in markdown/notebook docs.",
    "DOCX002": "Use MyST role syntax {doc}`...` instead of :doc:`...` in markdown/notebook docs.",
    "DOCX003": "Fix malformed role syntax such as :mod`...` / :doc`...` in markdown/notebook docs.",
    "DOCX004": "Use documentation cross-linking for public Deckard symbols (Config/Plugin/Mixin/Scorer classes, public functions, and public methods) instead of plain code formatting.",
    "DOCX005": "Use documentation cross-linking for framework/plugin references instead of plain code formatting.",
}

DOCS_ROLE_PATTERNS = {
    "DOCX001": re.compile(r":mod:`[^`]+`"),
    "DOCX002": re.compile(r":doc:`[^`]+`"),
    "DOCX003": re.compile(r"(?<!`):(?:mod|doc)`[^`]+`"),
}

INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")


@dataclass(frozen=True)
class DocsReferenceCatalog:
    symbol_tokens: frozenset[str]
    framework_tokens: frozenset[str]
    plugin_tokens: frozenset[str]
    canonical_literal_tokens: frozenset[str]


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


def _iter_docs_files(base: Path) -> Iterable[Path]:
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".ipynb"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if "build/" in rel or ".ipynb_checkpoints" in rel:
            continue
        yield path


def _module_from_path(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = list(rel.with_suffix("").parts)
    return ".".join(parts)


def _is_public_name(name: str) -> bool:
    return (not name.startswith("_")) or name == "__call__"


CANON_LITERAL_NAME_TOKENS = ("MODE", "STAGE", "ALIAS", "VALID")

# Inline docs tokens that are runtime field/column literals, not symbol refs.
DOCS_INLINE_LITERAL_EXCEPTIONS = frozenset(
    {
        "x",
        "y",
        "hue",
        "component",
        "event",
        "run",
        "load",
        "score",
        "train",
        "test",
        "val",
        "all",
        "attack",
        "attack-val",
        "pre-sample",
        "__call__",
    },
)


def _is_canon_literal_name(name: str) -> bool:
    upper_name = name.upper()
    return any(token in upper_name for token in CANON_LITERAL_NAME_TOKENS)


def _extract_string_literals(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        literals: set[str] = set()
        for child in node.elts:
            literals.update(_extract_string_literals(child))
        return literals

    if isinstance(node, ast.Dict):
        literals: set[str] = set()
        for child in (*node.keys, *node.values):
            literals.update(_extract_string_literals(child))
        return literals

    if isinstance(node, ast.Call):
        literals: set[str] = set()
        for arg in node.args:
            literals.update(_extract_string_literals(arg))
        for keyword in node.keywords:
            literals.update(_extract_string_literals(keyword.value))
        return literals

    return set()


def _expand_literal_variants(token: str) -> set[str]:
    base = str(token).strip()
    if not base:
        return set()

    lower = base.lower()
    variants = {
        base,
        lower,
        lower.replace("_", "-"),
        lower.replace(" ", "-"),
        lower.replace("-", "_"),
        lower.replace("-", " "),
        lower.replace("_", " "),
    }
    return {variant.strip() for variant in variants if variant.strip()}


def build_docs_reference_catalog(
    source_scope: Path = DECKARD_DIR,
) -> DocsReferenceCatalog:
    """Build source-derived symbol and framework/plugin token sets for docs checks."""
    symbol_tokens: set[str] = set()
    canonical_literal_tokens: set[str] = set()

    for path in _iter_python_files(source_scope):
        module_name = _module_from_path(path)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path.as_posix())

        if path.name == "canon.py":
            for node in tree.body:
                target_names: list[str] = []
                value_node: ast.AST | None = None

                if isinstance(node, ast.Assign):
                    value_node = node.value
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            target_names.append(target.id)
                elif isinstance(node, ast.AnnAssign):
                    value_node = node.value
                    if isinstance(node.target, ast.Name):
                        target_names.append(node.target.id)

                if not target_names or value_node is None:
                    continue
                if not any(_is_canon_literal_name(name) for name in target_names):
                    continue

                for literal in _extract_string_literals(value_node):
                    canonical_literal_tokens.update(_expand_literal_variants(literal))

        for node in tree.body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ) and _is_public_name(node.name):
                symbol_tokens.add(node.name)

            if not isinstance(node, ast.ClassDef):
                continue

            class_name = node.name
            if not _is_public_name(class_name):
                continue

            tracked_class = (
                class_name.endswith(("Config", "Plugin", "Mixin"))
                or "Scorer" in class_name
            )
            if tracked_class:
                symbol_tokens.add(class_name)

            for child in node.body:
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ) and _is_public_name(child.name):
                    symbol_tokens.add(child.name)
                    symbol_tokens.add(f"{class_name}.{child.name}")
                    symbol_tokens.add(f"{module_name}.{class_name}.{child.name}")

    frameworks_dir = DECKARD_DIR / "frameworks"
    framework_tokens = {
        p.name.lower()
        for p in frameworks_dir.iterdir()
        if p.is_dir() and not p.name.startswith("__")
    }

    plugins_dir = DECKARD_DIR / "plugins"
    plugin_tokens = {
        p.name.lower()
        for p in plugins_dir.iterdir()
        if p.is_dir() and not p.name.startswith("__")
    }

    canonical_literal_tokens.update(DOCS_INLINE_LITERAL_EXCEPTIONS)

    return DocsReferenceCatalog(
        symbol_tokens=frozenset(symbol_tokens),
        framework_tokens=frozenset(framework_tokens),
        plugin_tokens=frozenset(plugin_tokens),
        canonical_literal_tokens=frozenset(canonical_literal_tokens),
    )


def _iter_docs_lines(path: Path) -> list[tuple[int, str]]:
    """Return line-like tuples for markdown docs and notebook markdown cells."""
    if path.suffix == ".md":
        return [
            (i, line)
            for i, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            )
        ]

    if path.suffix == ".ipynb":
        content = json.loads(path.read_text(encoding="utf-8"))
        entries: list[tuple[int, str]] = []
        line_no = 1
        for cell in content.get("cells", []):
            if cell.get("cell_type") == "markdown":
                source = cell.get("source", [])
                if isinstance(source, str):
                    source_lines = source.splitlines()
                else:
                    source_lines = "".join(source).splitlines()
                for line in source_lines:
                    entries.append((line_no, line))
                    line_no += 1
            else:
                line_no += 1
        return entries

    return []


def _has_docs_crosslink(line: str, token: str) -> bool:
    escaped = re.escape(token)
    role_pattern = re.compile(
        rf"\{{(?:class|func|meth|mod|doc)\}}`[^`]*\b{escaped}\b[^`]*`",
    )
    markdown_link_pattern = re.compile(rf"\[[^\]]*\b{escaped}\b[^\]]*\]\([^\)]+\)")
    return bool(role_pattern.search(line) or markdown_link_pattern.search(line))


def validate_docs_file(path: Path, catalog: DocsReferenceCatalog) -> list[Violation]:
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()

    violations: list[Violation] = []
    lines = _iter_docs_lines(path)
    skip_plain_ref_checks = rel in {
        "docs/overview/changelog.md",
        "docs/developers/refactor_plan.md",
    }
    in_code_fence = False

    for line_no, line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence

        for code, pattern in DOCS_ROLE_PATTERNS.items():
            if pattern.search(line):
                violations.append(
                    Violation(
                        rel,
                        line_no,
                        code,
                        RULE_ANNOTATIONS[code],
                    ),
                )

        if skip_plain_ref_checks or in_code_fence:
            continue

        inline_tokens = [
            m.group(1).strip() for m in INLINE_CODE_PATTERN.finditer(line)
        ]
        docx004_emitted = False
        docx005_emitted = False
        for token in inline_tokens:
            if (
                token in catalog.canonical_literal_tokens
                or token.lower() in catalog.canonical_literal_tokens
            ):
                continue

            if (
                not docx004_emitted
                and token in catalog.symbol_tokens
                and not _has_docs_crosslink(line, token)
            ):
                violations.append(
                    Violation(
                        rel,
                        line_no,
                        "DOCX004",
                        RULE_ANNOTATIONS["DOCX004"],
                    ),
                )
                docx004_emitted = True
                continue

            lower_token = token.lower()
            if (
                not docx005_emitted
                and (
                    lower_token in catalog.framework_tokens
                    or lower_token in catalog.plugin_tokens
                )
            ) and not _has_docs_crosslink(line, token):
                violations.append(
                    Violation(
                        rel,
                        line_no,
                        "DOCX005",
                        RULE_ANNOTATIONS["DOCX005"],
                    ),
                )
                docx005_emitted = True

    return violations


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
    return class_name.endswith(suffixes) or any(
        token in class_name for token in tokens
    )


def _class_field_annotations(node: ast.ClassDef) -> list[ast.AnnAssign]:
    anns: list[ast.AnnAssign] = []
    for child in node.body:
        if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            anns.append(child)
    return anns


def _is_dataclass_decorated(node: ast.ClassDef) -> bool:
    decorators = {_decorator_name(d) for d in node.decorator_list}
    return "dataclass" in decorators


def _annotation_is_classvar(annotation: ast.AST | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id == "ClassVar"
    if isinstance(annotation, ast.Attribute):
        return annotation.attr == "ClassVar"
    if isinstance(annotation, ast.Subscript):
        return _annotation_is_classvar(annotation.value)
    return False


def _field_init_disabled(value: ast.AST | None) -> bool:
    if not isinstance(value, ast.Call):
        return False
    func_name = _decorator_name(value.func)
    if func_name != "field":
        return False
    for kw in value.keywords:
        if kw.arg == "init":
            return isinstance(kw.value, ast.Constant) and kw.value.value is False
    return False


def _field_call(value: ast.AST | None) -> ast.Call | None:
    if not isinstance(value, ast.Call):
        return None
    func_name = _decorator_name(value.func)
    if func_name != "field":
        return None
    return value


def _field_has_metadata(value: ast.AST | None) -> bool:
    call = _field_call(value)
    if call is None:
        return False
    for kw in call.keywords:
        if kw.arg == "metadata":
            return not (
                isinstance(kw.value, ast.Constant) and kw.value.value is None
            )
    return False


def _field_kw_bool(value: ast.AST | None, keyword: str) -> bool | None:
    call = _field_call(value)
    if call is None:
        return None
    for kw in call.keywords:
        if kw.arg == keyword:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                return bool(kw.value.value)
            return None
    return None


def _field_repr_disabled(value: ast.AST | None) -> bool:
    repr_kw = _field_kw_bool(value, "repr")
    return repr_kw is False


def _field_default_string(value: ast.AST | None) -> str | None:
    call = _field_call(value)
    if call is None:
        return None
    for kw in call.keywords:
        if kw.arg == "default":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value
            return None
    return None


def _canonical_object_path(path: Path, class_name: str) -> str:
    try:
        module_name = _module_from_path(path)
    except ValueError:
        module_name = path.with_suffix("").name
    return f"{module_name}.{class_name}"


def _looks_like_runtime_field(field_name: str) -> bool:
    if field_name == "_target_":
        return False
    if field_name in RUNTIME_FIELD_NAMES:
        return True
    if field_name.startswith("_") and not field_name.startswith("__"):
        return True
    return False


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
    enforce_class_contracts: bool = True,
    strict_docs_types: bool = False,
    require_attributes_sections: bool = False,
    enforce_runtime_init_params: bool = False,
    enforce_field_metadata: bool = False,
    enforce_runtime_repr: bool = False,
    enforce_target_field: bool = False,
) -> list[Violation]:
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=rel)
    violations: list[Violation] = []

    # CLS005: default mode policy for RST tokens across all docstrings.
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
                    "CLS005",
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

        if enforce_class_contracts:
            if _is_public_name(class_name):
                field_anns = _class_field_annotations(node)
                # Require dataclass when public classes define concrete annotated defaults,
                # not when they only declare type hints for static analysis.
                has_concrete_field = any(ann.value is not None for ann in field_anns)
                # CLS001: concrete field defaults in public classes require dataclass semantics.
                if has_concrete_field and "dataclass" not in decorators:
                    violations.append(
                        Violation(
                            rel,
                            node.lineno,
                            "CLS001",
                            f"{class_name} defines annotated fields and must be a dataclass",
                        ),
                    )

                public_methods = [
                    c
                    for c in node.body
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and _is_public_method(c)
                ]
                protected_hook_methods = [
                    c
                    for c in node.body
                    if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and c.name in PROTECTED_MIXIN_HOOK_METHODS
                ]

                class_doc = ast.get_docstring(node) or ""
                # CLS004/CLS005: class docstring presence + token policy.
                if not class_doc.strip():
                    violations.append(
                        Violation(
                            rel,
                            node.lineno,
                            "CLS004",
                            f"{class_name} missing class docstring",
                        ),
                    )

                # CLS006: public class methods must be documented.
                for fn in public_methods:
                    doc = ast.get_docstring(fn) or ""
                    if not doc.strip():
                        violations.append(
                            Violation(
                                rel,
                                fn.lineno,
                                "CLS006",
                                f"{class_name}.{fn.name} missing public docstring",
                            ),
                        )

                # CLS007: selected protected hook methods must also be documented.
                for fn in protected_hook_methods:
                    doc = ast.get_docstring(fn) or ""
                    if not doc.strip():
                        violations.append(
                            Violation(
                                rel,
                                fn.lineno,
                                "CLS007",
                                f"{class_name}.{fn.name} missing protected hook docstring",
                            ),
                        )

        methods = [
            c
            for c in node.body
            if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if enforce_runtime_init_params and _is_dataclass_decorated(node):
            for ann in _class_field_annotations(node):
                field_name = ann.target.id
                if _annotation_is_classvar(ann.annotation):
                    continue
                if not _looks_like_runtime_field(field_name):
                    continue
                if _field_init_disabled(ann.value):
                    continue
                violations.append(
                    Violation(
                        rel,
                        ann.lineno,
                        "CFG007",
                        (
                            f"{class_name}.{field_name} looks like a runtime field "
                            "and must set field(init=False)"
                        ),
                    ),
                )

        if enforce_field_metadata and _is_dataclass_decorated(node):
            for ann in _class_field_annotations(node):
                field_name = ann.target.id
                if _annotation_is_classvar(ann.annotation):
                    continue
                if _field_call(ann.value) is None:
                    continue
                if _field_has_metadata(ann.value):
                    continue
                violations.append(
                    Violation(
                        rel,
                        ann.lineno,
                        "CFG008",
                        (
                            f"{class_name}.{field_name} uses field() and must set "
                            "field(metadata={...})"
                        ),
                    ),
                )

        if enforce_runtime_repr and _is_dataclass_decorated(node):
            for ann in _class_field_annotations(node):
                field_name = ann.target.id
                if _annotation_is_classvar(ann.annotation):
                    continue
                if field_name == "_target_":
                    continue
                if _field_call(ann.value) is None:
                    continue
                if not _field_init_disabled(ann.value):
                    continue
                if _field_repr_disabled(ann.value):
                    continue
                violations.append(
                    Violation(
                        rel,
                        ann.lineno,
                        "CFG009",
                        (
                            f"{class_name}.{field_name} uses field(init=False) and must also set "
                            "field(repr=False) to match fingerprint/to_yaml behavior"
                        ),
                    ),
                )

        if enforce_target_field and _is_dataclass_decorated(node):
            for ann in _class_field_annotations(node):
                field_name = ann.target.id
                if field_name != "_target_":
                    continue
                if _annotation_is_classvar(ann.annotation):
                    continue

                canonical_target = _canonical_object_path(path, class_name)
                call = _field_call(ann.value)
                if call is None:
                    violations.append(
                        Violation(
                            rel,
                            ann.lineno,
                            "CFG010",
                            (
                                f"{class_name}._target_ must use field(default=\"{canonical_target}\") "
                                "with init/repr enabled"
                            ),
                        ),
                    )
                    continue

                init_kw = _field_kw_bool(ann.value, "init")
                repr_kw = _field_kw_bool(ann.value, "repr")
                default_value = _field_default_string(ann.value)
                if (
                    init_kw is False
                    or repr_kw is False
                    or default_value != canonical_target
                ):
                    violations.append(
                        Violation(
                            rel,
                            ann.lineno,
                            "CFG010",
                            (
                                f"{class_name}._target_ must default to \"{canonical_target}\" "
                                "with init=True and repr=True"
                            ),
                        ),
                    )

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
            class_is_canonical_runtime = class_name.endswith(
                ("Config", "Mixin", "Plugin"),
            )
            for fn in methods:
                if not _is_public_method(fn):
                    continue

                if class_is_canonical_runtime:
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
                    if not (enforce_class_contracts and _is_public_name(class_name)):
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
                        class_is_canonical_runtime
                        and fn.name != "__init__"
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
                        class_is_canonical_runtime
                        and
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
                        if not class_is_canonical_runtime:
                            continue
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
    enforce_class_contracts: bool = True,
    strict_docs_types: bool = False,
    require_attributes_sections: bool = False,
    enforce_runtime_init_params: bool = False,
    enforce_field_metadata: bool = False,
    enforce_runtime_repr: bool = False,
    enforce_target_field: bool = False,
) -> list[Violation]:
    candidate = Path(scope)
    base = candidate if candidate.is_absolute() else ROOT / scope
    violations: list[Violation] = []
    if base.is_file() and base.suffix == ".py":
        violations.extend(
            validate_file(
                base,
                enforce_class_contracts=enforce_class_contracts,
                strict_docs_types=strict_docs_types,
                require_attributes_sections=require_attributes_sections,
                enforce_runtime_init_params=enforce_runtime_init_params,
                enforce_field_metadata=enforce_field_metadata,
                enforce_runtime_repr=enforce_runtime_repr,
                enforce_target_field=enforce_target_field,
            ),
        )
        return sorted(violations, key=lambda v: (v.path, v.line, v.code, v.message))
    for path in _iter_python_files(base):
        violations.extend(
            validate_file(
                path,
                enforce_class_contracts=enforce_class_contracts,
                strict_docs_types=strict_docs_types,
                require_attributes_sections=require_attributes_sections,
                enforce_runtime_init_params=enforce_runtime_init_params,
                enforce_field_metadata=enforce_field_metadata,
                enforce_runtime_repr=enforce_runtime_repr,
                enforce_target_field=enforce_target_field,
            ),
        )
    return sorted(violations, key=lambda v: (v.path, v.line, v.code, v.message))


def collect_docs_violations(scope: str) -> list[Violation]:
    candidate = Path(scope)
    base = candidate if candidate.is_absolute() else ROOT / scope
    violations: list[Violation] = []
    catalog = build_docs_reference_catalog()

    if base.is_file() and base.suffix in {".md", ".ipynb"}:
        violations.extend(validate_docs_file(base, catalog))
        return sorted(violations, key=lambda v: (v.path, v.line, v.code, v.message))

    for path in _iter_docs_files(base):
        violations.extend(validate_docs_file(path, catalog))
    return sorted(violations, key=lambda v: (v.path, v.line, v.code, v.message))


def collect_docs_audit(scope: str) -> list[dict[str, object]]:
    """Return per-file docs enforcement coverage and violation metadata."""
    candidate = Path(scope)
    base = candidate if candidate.is_absolute() else ROOT / scope
    catalog = build_docs_reference_catalog()
    entries: list[dict[str, object]] = []

    if base.is_file() and base.suffix in {".md", ".ipynb"}:
        paths = [base]
    else:
        paths = sorted(_iter_docs_files(base), key=lambda p: p.as_posix())

    for path in paths:
        violations = validate_docs_file(path, catalog)
        rule_counts = Counter(v.code for v in violations)
        lines_checked = len(_iter_docs_lines(path))
        rel = path.relative_to(ROOT).as_posix()
        entries.append(
            {
                "path": rel,
                "lines_checked": lines_checked,
                "violation_count": len(violations),
                "violation_codes": dict(sorted(rule_counts.items())),
            },
        )

    return entries


def _write_docs_audit_report(
    report_path: str,
    docs_scopes: list[str],
    entries: list[dict[str, object]],
) -> Path:
    target = Path(report_path)
    if not target.is_absolute():
        target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)

    files_with_violations = sum(
        1 for entry in entries if int(entry.get("violation_count", 0)) > 0
    )
    payload = {
        "docs_scopes": docs_scopes,
        "summary": {
            "files_checked": len(entries),
            "files_with_violations": files_with_violations,
            "total_violations": sum(
                int(entry.get("violation_count", 0)) for entry in entries
            ),
        },
        "files": entries,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _parse_docs_scopes(raw_scope: str) -> list[str]:
    """Parse comma-separated docs scopes; supports 'none' to disable docs checks."""
    value = str(raw_scope).strip()
    if value.lower() == "none":
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


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
        "--enforce-class-contracts",
        dest="enforce_class_contracts",
        action="store_true",
        help="Enforce public-class contract rules (CLS001/CLS004/CLS006/CLS007) (enabled by default)",
    )
    parser.add_argument(
        "--no-enforce-class-contracts",
        dest="enforce_class_contracts",
        action="store_false",
        help="Disable public-class contract enforcement rules",
    )
    parser.add_argument(
        "--strict-docs-types",
        dest="strict_docs_types",
        action="store_true",
        help="Enable strict public docstring/type-annotation checks (enabled by default)",
    )
    parser.add_argument(
        "--no-strict-docs-types",
        dest="strict_docs_types",
        action="store_false",
        help="Disable strict public docstring/type-annotation checks",
    )
    parser.add_argument(
        "--require-attributes-sections",
        dest="require_attributes_sections",
        action="store_true",
        help="Require Google-style 'Attributes:' sections for Config/Mixin/Plugin and related runtime classes (enabled by default)",
    )
    parser.add_argument(
        "--no-require-attributes-sections",
        dest="require_attributes_sections",
        action="store_false",
        help="Disable Attributes section enforcement",
    )
    parser.add_argument(
        "--enforce-runtime-init-params",
        dest="enforce_runtime_init_params",
        action="store_true",
        help=(
            "Enforce that runtime-looking dataclass fields (for example private "
            "state fields) are marked field(init=False) (enabled by default)"
        ),
    )
    parser.add_argument(
        "--no-enforce-runtime-init-params",
        dest="enforce_runtime_init_params",
        action="store_false",
        help="Disable runtime init=False enforcement",
    )
    parser.add_argument(
        "--enforce-field-metadata",
        dest="enforce_field_metadata",
        action="store_true",
        help=(
            "Enforce that dataclass field() declarations explicitly provide "
            "metadata={...} (enabled by default)"
        ),
    )
    parser.add_argument(
        "--no-enforce-field-metadata",
        dest="enforce_field_metadata",
        action="store_false",
        help="Disable dataclass field metadata enforcement",
    )
    parser.add_argument(
        "--enforce-runtime-repr",
        dest="enforce_runtime_repr",
        action="store_true",
        help=(
            "Enforce that dataclass field(init=False) declarations also set "
            "repr=False so repr matches BaseConfig fingerprint/YAML behavior "
            "for non-_target_ runtime fields (enabled by default)"
        ),
    )
    parser.add_argument(
        "--no-enforce-runtime-repr",
        dest="enforce_runtime_repr",
        action="store_false",
        help="Disable runtime repr=False enforcement",
    )
    parser.add_argument(
        "--enforce-target-field",
        dest="enforce_target_field",
        action="store_true",
        help=(
            "Enforce that dataclass _target_ fields preserve init/repr visibility "
            "and default to the canonical Deckard object path (enabled by default)"
        ),
    )
    parser.add_argument(
        "--no-enforce-target-field",
        dest="enforce_target_field",
        action="store_false",
        help="Disable canonical _target_ field enforcement",
    )
    parser.add_argument(
        "--docs-scope",
        default=",".join(DEFAULT_DOCS_SCOPES),
        help=(
            "Docs scope to validate markdown/notebook cross-reference syntax "
            "(comma-separated). "
            f"Default: {', '.join(DEFAULT_DOCS_SCOPES)}. Use 'none' to disable docs checks."
        ),
    )
    parser.add_argument(
        "--docs-audit-report",
        default="",
        help=(
            "Optional JSON output path for per-file docs audit coverage and "
            "violation counts. Can be relative to repository root."
        ),
    )
    parser.set_defaults(
        enforce_class_contracts=True,
        strict_docs_types=True,
        require_attributes_sections=True,
        enforce_runtime_init_params=True,
        enforce_field_metadata=True,
        enforce_runtime_repr=True,
        enforce_target_field=True,
    )
    args = parser.parse_args()

    violations = collect_violations(
        args.scope,
        enforce_class_contracts=args.enforce_class_contracts,
        strict_docs_types=args.strict_docs_types,
        require_attributes_sections=args.require_attributes_sections,
        enforce_runtime_init_params=args.enforce_runtime_init_params,
        enforce_field_metadata=args.enforce_field_metadata,
        enforce_runtime_repr=args.enforce_runtime_repr,
        enforce_target_field=args.enforce_target_field,
    )
    docs_scopes = _parse_docs_scopes(args.docs_scope)
    if docs_scopes:
        for docs_scope in docs_scopes:
            violations.extend(collect_docs_violations(docs_scope))
        violations = sorted(
            violations,
            key=lambda v: (v.path, v.line, v.code, v.message),
        )

    if args.docs_audit_report and docs_scopes:
        audit_entries: list[dict[str, object]] = []
        for docs_scope in docs_scopes:
            audit_entries.extend(collect_docs_audit(docs_scope))
        audit_entries = sorted(
            audit_entries,
            key=lambda item: str(item.get("path", "")),
        )
        report_path = _write_docs_audit_report(
            args.docs_audit_report,
            docs_scopes,
            audit_entries,
        )
        print(
            "Wrote docs audit report to "
            f"{report_path.relative_to(ROOT).as_posix() if report_path.is_relative_to(ROOT) else report_path}",
        )
    if violations:
        for violation in violations:
            print(violation.format())
        if docs_scopes:
            docs_scope_label = ", ".join(docs_scopes)
            print(
                f"\nFound {len(violations)} enforcement violations in {args.scope} and {docs_scope_label}",
            )
        else:
            print(f"\nFound {len(violations)} enforcement violations in {args.scope}")
        return 1

    if docs_scopes:
        docs_scope_label = ", ".join(docs_scopes)
        print(
            f"No enforcement violations found in {args.scope} and {docs_scope_label}",
        )
    else:
        print(f"No enforcement violations found in {args.scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
