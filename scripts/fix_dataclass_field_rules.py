#!/usr/bin/env python3
"""Autofix CFG007/CFG008/CFG009 dataclass field violations.

Updates dataclass field declarations so runtime fields use ``init=False`` and all
``field(...)`` declarations include ``metadata``. Also ensures non-``_target_``
``field(init=False)`` declarations include ``repr=False``.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

from repository_enforcement import (
    ROOT,
    _annotation_is_classvar,
    _field_call,
    _field_has_metadata,
    _field_init_disabled,
    _field_repr_disabled,
    _is_dataclass_decorated,
    _iter_python_files,
    _looks_like_runtime_field,
)

DEFAULT_METADATA_HELP = "TODO: document field."


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    replacement: str


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    running = 0
    for line in source.splitlines(keepends=True):
        running += len(line)
        offsets.append(running)
    return offsets


def _to_offset(offsets: list[int], lineno: int, col: int) -> int:
    return offsets[lineno - 1] + col


def _node_end(node: ast.AST) -> tuple[int, int]:
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    assert end_lineno is not None
    assert end_col_offset is not None
    return end_lineno, end_col_offset


def _replace_slice(source: str, edits: list[Edit]) -> str:
    updated = source
    for edit in sorted(edits, key=lambda item: item.start, reverse=True):
        updated = updated[: edit.start] + edit.replacement + updated[edit.end :]
    return updated


def _dict_has_help_key(node: ast.AST) -> bool:
    if not isinstance(node, ast.Dict):
        return False
    for key in node.keys:
        if isinstance(key, ast.Constant) and key.value == "help":
            return True
    return False


def _metadata_value(field_name: str) -> ast.Dict:
    return ast.Dict(
        keys=[ast.Constant(value="help")],
        values=[ast.Constant(value=f"TODO: document {field_name}.")],
    )


def _build_field_call(
    ann: ast.AnnAssign,
    *,
    add_init_false: bool,
    add_metadata: bool,
    add_repr_false: bool,
) -> ast.Call:
    value = ann.value
    assert isinstance(ann.target, ast.Name)
    assert value is not None
    field_name = ann.target.id
    call = _field_call(value)
    if call is None:
        keywords: list[ast.keyword] = [ast.keyword(arg="default", value=value)]
    else:
        keywords = [ast.keyword(arg=kw.arg, value=kw.value) for kw in call.keywords]

    existing = {kw.arg for kw in keywords}
    if add_init_false and "init" not in existing:
        keywords.append(ast.keyword(arg="init", value=ast.Constant(value=False)))
    if add_repr_false and "repr" not in existing:
        keywords.append(ast.keyword(arg="repr", value=ast.Constant(value=False)))
    if add_metadata and "metadata" not in existing:
        keywords.append(
            ast.keyword(arg="metadata", value=_metadata_value(field_name)),
        )

    if call is not None:
        args = list(call.args)
        func = call.func
    else:
        args = []
        func = ast.Name(id="field", ctx=ast.Load())

    return ast.Call(func=func, args=args, keywords=keywords)


def _field_import_present(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
            return any(alias.name == "field" for alias in node.names)
    return False


def _ensure_field_import(source: str, tree: ast.Module) -> str:
    if _field_import_present(tree):
        return source

    offsets = _line_offsets(source)
    dataclasses_from = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "dataclasses"
        ),
        None,
    )
    if dataclasses_from is not None:
        start = _to_offset(
            offsets,
            dataclasses_from.lineno,
            dataclasses_from.col_offset,
        )
        end_lineno, end_col_offset = _node_end(dataclasses_from)
        end = _to_offset(
            offsets,
            end_lineno,
            end_col_offset,
        )
        names = [alias.name for alias in dataclasses_from.names]
        names.append("field")
        replacement = (
            f"from dataclasses import {', '.join(sorted(dict.fromkeys(names)))}"
        )
        return source[:start] + replacement + source[end:]

    insert_at = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(
            tree.body[0].value,
            ast.Constant,
        )
        and isinstance(tree.body[0].value.value, str)
    ):
        end_lineno, end_col_offset = _node_end(tree.body[0])
        insert_at = _to_offset(
            offsets,
            end_lineno,
            end_col_offset,
        )
        prefix = "\n\n"
    else:
        prefix = ""
    return (
        source[:insert_at]
        + f"{prefix}from dataclasses import field\n"
        + source[insert_at:]
    )


def _fix_source(
    source: str,
    *,
    fix_cfg007: bool,
    fix_cfg008: bool,
    fix_cfg009: bool,
) -> tuple[str, bool]:
    tree = ast.parse(source)
    offsets = _line_offsets(source)
    edits: list[Edit] = []
    need_field_import = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or not _is_dataclass_decorated(node):
            continue
        for ann in [child for child in node.body if isinstance(child, ast.AnnAssign)]:
            if not isinstance(ann.target, ast.Name):
                continue
            if ann.value is None:
                continue
            if _annotation_is_classvar(ann.annotation):
                continue

            add_init_false = False
            add_metadata = False
            add_repr_false = False
            field_name = ann.target.id
            call = _field_call(ann.value)

            if fix_cfg007 and _looks_like_runtime_field(field_name):
                add_init_false = not _field_init_disabled(ann.value)
            if fix_cfg008 and (call is not None or add_init_false):
                add_metadata = not _field_has_metadata(ann.value)
            if (
                fix_cfg009
                and field_name != "_target_"
                and _field_init_disabled(ann.value)
            ):
                add_repr_false = not _field_repr_disabled(ann.value)

            if not add_init_false and not add_metadata and not add_repr_false:
                continue

            replacement = ast.unparse(
                _build_field_call(
                    ann,
                    add_init_false=add_init_false,
                    add_metadata=add_metadata,
                    add_repr_false=add_repr_false,
                ),
            )
            edits.append(
                Edit(
                    start=_to_offset(offsets, ann.value.lineno, ann.value.col_offset),
                    end=_to_offset(offsets, *_node_end(ann.value)),
                    replacement=replacement,
                ),
            )
            if call is None:
                need_field_import = True

    if not edits:
        return source, False

    updated = _replace_slice(source, edits)
    if need_field_import:
        updated = _ensure_field_import(updated, ast.parse(updated))
    return updated, updated != source


def _iter_targets(scope: str) -> list[Path]:
    target = (ROOT / scope).resolve() if not Path(scope).is_absolute() else Path(scope)
    if target.is_file():
        return [target]
    return list(_iter_python_files(target))


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="deckard", help="File or directory to fix")
    parser.add_argument("--fix-cfg007", action="store_true")
    parser.add_argument("--fix-cfg008", action="store_true")
    parser.add_argument("--fix-cfg009", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    any_specific = args.fix_cfg007 or args.fix_cfg008 or args.fix_cfg009
    fix_cfg007 = args.fix_cfg007 or not any_specific
    fix_cfg008 = args.fix_cfg008 or not any_specific
    fix_cfg009 = args.fix_cfg009 or not any_specific

    changed_files: list[str] = []
    for path in _iter_targets(args.scope):
        source = path.read_text(encoding="utf-8")
        updated, changed = _fix_source(
            source,
            fix_cfg007=fix_cfg007,
            fix_cfg008=fix_cfg008,
            fix_cfg009=fix_cfg009,
        )
        if not changed:
            continue
        rel = _display_path(path)
        changed_files.append(rel)
        if not args.dry_run:
            path.write_text(updated, encoding="utf-8")

    verb = "Would update" if args.dry_run else "Updated"
    print(f"{verb} {len(changed_files)} Python files")
    for rel in changed_files:
        print(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
