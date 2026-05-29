#!/usr/bin/env python3
"""Bulk-fix DOCX004/DOCX005 docs violations.

Converts plain inline-code references in docs/*.md and docs/*.ipynb markdown cells
into markdown links using source-derived symbol/framework/plugin catalogs.
"""

from __future__ import annotations

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from repository_enforcement import build_docs_reference_catalog

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
API_MODULES = DOCS / "api" / "modules"
EXT_INDEX = DOCS / "overview" / "extensions" / "index"
DEV_INDEX = DOCS / "developers" / "index"
DECKARD = ROOT / "deckard"
INLINE_CODE_PATTERN = re.compile(r"`([^`]+)`")
BROKEN_LINK_TARGET = "TODO-BROKEN-LINK"


def _module_from_path(path: Path) -> str:
    rel = path.relative_to(ROOT)
    return ".".join(rel.with_suffix("").parts)


def _is_public_name(name: str) -> bool:
    return (not name.startswith("_")) or name == "__call__"


def _resolve_api_target(*slugs: str) -> Path:
    for slug in slugs:
        flat = DOCS / "api" / f"{slug}.md"
        nested = DOCS / "api" / slug / "index.md"
        if flat.exists():
            return DOCS / "api" / slug
        if nested.exists():
            return DOCS / "api" / slug / "index"
    return API_MODULES


def _resolve_developer_target(*slugs: str) -> Path:
    for slug in slugs:
        flat = DOCS / "developers" / f"{slug}.md"
        nested = DOCS / "developers" / slug / "index.md"
        if flat.exists():
            return DOCS / "developers" / slug
        if nested.exists():
            return DOCS / "developers" / slug / "index"
    return DEV_INDEX


def _api_target_for_source(path: Path) -> Path:
    rel_parts = path.relative_to(DECKARD).with_suffix("").parts
    if not rel_parts:
        return API_MODULES

    root = rel_parts[0]
    second = rel_parts[1] if len(rel_parts) > 1 else ""

    if root == "data" and second in {"sample", "pipeline"}:
        return _resolve_api_target(second)
    if root == "model" and second in {"train", "defend"}:
        return _resolve_api_target(second)
    if root == "frameworks":
        if second and second != "__init__":
            return _resolve_api_target(second, f"frameworks/{second}", "frameworks")
        return _resolve_api_target("frameworks")
    if root == "plugins":
        if second and second != "__init__":
            return _resolve_api_target(f"plugins/{second}", second, "plugins")
        return _resolve_api_target("plugins")

    return _resolve_api_target(root)


def _build_symbol_link_targets(catalog) -> dict[str, Path]:
    token_pages: dict[str, set[Path]] = defaultdict(set)

    for path in DECKARD.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path.as_posix())
        module_name = _module_from_path(path)
        page = _api_target_for_source(path)

        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and _is_public_name(node.name):
                token_pages[node.name].add(page)

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
                token_pages[class_name].add(page)

            for child in node.body:
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and _is_public_name(child.name):
                    token_pages[child.name].add(page)
                    token_pages[f"{class_name}.{child.name}"].add(page)
                    token_pages[f"{module_name}.{class_name}.{child.name}"].add(page)

    resolved: dict[str, Path] = {}
    for token in catalog.symbol_tokens:
        pages = token_pages.get(token)
        if not pages:
            continue
        if len(pages) == 1:
            resolved[token] = next(iter(pages))

    return resolved


def _build_extension_link_targets(catalog) -> dict[str, Path]:
    ext_dir = DOCS / "overview" / "extensions"
    mapping: dict[str, Path] = {}

    for token in catalog.framework_tokens:
        candidates = [
            _resolve_api_target(token, f"frameworks/{token}"),
            _resolve_developer_target(f"extensions/{token}", "extensions"),
            ext_dir / token,
            EXT_INDEX,
        ]
        for candidate in candidates:
            if (
                (candidate.with_suffix(".md")).exists()
                or (candidate / "index.md").exists()
                or candidate == EXT_INDEX
            ):
                mapping[token] = candidate
                break

    for token in catalog.plugin_tokens:
        candidates = [
            _resolve_api_target(f"plugins/{token}", "plugins"),
            _resolve_developer_target(f"extensions/{token}", "extensions"),
            ext_dir / token,
            EXT_INDEX,
        ]
        for candidate in candidates:
            if (
                (candidate.with_suffix(".md")).exists()
                or (candidate / "index.md").exists()
                or candidate == EXT_INDEX
            ):
                mapping[token] = candidate
                break

    return mapping


def _todo_link(token: str, kind: str) -> str:
    comment = (
        f"<!-- TODO(docs): map '{token}' to a domain-specific {kind} docs page -->"
    )
    return f"[{token}]({BROKEN_LINK_TARGET}) {comment}"


def _rel_link(from_file: Path, target_no_ext: Path) -> str:
    return Path(
        os.path.relpath(target_no_ext.as_posix(), from_file.parent.as_posix()),
    ).as_posix()


def _rewrite_lines(
    lines: list[str],
    from_file: Path,
    catalog,
    symbol_targets: dict[str, Path],
    extension_targets: dict[str, Path],
) -> tuple[list[str], bool]:
    out: list[str] = []
    changed = False
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue

        def _replace(match: re.Match[str]) -> str:
            nonlocal changed
            token = match.group(1).strip()
            lowered = token.lower()
            if token in catalog.symbol_tokens:
                changed = True
                target = symbol_targets.get(token)
                if target is None:
                    return _todo_link(token, "API")
                return f"[{token}]({_rel_link(from_file, target)})"
            if lowered in catalog.framework_tokens or lowered in catalog.plugin_tokens:
                changed = True
                target = extension_targets.get(lowered)
                if target is None:
                    return _todo_link(token, "extension")
                return f"[{token}]({_rel_link(from_file, target)})"
            return match.group(0)

        out.append(INLINE_CODE_PATTERN.sub(_replace, line))

    return out, changed


def _fix_markdown(
    path: Path,
    catalog,
    symbol_targets: dict[str, Path],
    extension_targets: dict[str, Path],
) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines, changed = _rewrite_lines(
        lines, path, catalog, symbol_targets, extension_targets
    )
    if changed:
        path.write_text("".join(new_lines), encoding="utf-8")
    return changed


def _fix_notebook(
    path: Path,
    catalog,
    symbol_targets: dict[str, Path],
    extension_targets: dict[str, Path],
) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", [])
        if isinstance(src, str):
            lines = src.splitlines(keepends=True)
        else:
            lines = list(src)
        new_lines, cell_changed = _rewrite_lines(
            lines, path, catalog, symbol_targets, extension_targets
        )
        if cell_changed:
            cell["source"] = new_lines
            changed = True

    if changed:
        path.write_text(
            json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    return changed


def main() -> int:
    catalog = build_docs_reference_catalog()
    symbol_targets = _build_symbol_link_targets(catalog)
    extension_targets = _build_extension_link_targets(catalog)
    changed_files: list[str] = []

    for path in DOCS.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".ipynb"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if "build/" in rel or ".ipynb_checkpoints" in rel:
            continue

        changed = False
        if path.suffix == ".md":
            changed = _fix_markdown(path, catalog, symbol_targets, extension_targets)
        else:
            changed = _fix_notebook(path, catalog, symbol_targets, extension_targets)

        if changed:
            changed_files.append(rel)

    print(f"Updated {len(changed_files)} docs files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
