#!/usr/bin/env python3
"""Synchronize Deckard's own version across project files.

Source of truth: the newest version header in docs/overview/changelog.md.

Accepted changelog header forms:
- ## 0.98.3
- ## .98.3 (normalized to 0.98.3)

Targets:
- pyproject.toml -> [project].version = "X.Y.Z"
- docs/conf.py -> release = "X.Y.Z"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DOCS_CONF = ROOT / "docs" / "conf.py"
CHANGELOG = ROOT / "docs" / "overview" / "changelog.md"


def read_changelog_version() -> str:
    text = CHANGELOG.read_text(encoding="utf-8")
    match = re.search(
        r"^##\s+([0-9]+\.[0-9]+\.[0-9]+|\.[0-9]+\.[0-9]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(
            "Could not read version header from docs/overview/changelog.md",
        )

    raw = match.group(1)
    if raw.startswith("."):
        major_minor = raw[1:]
        major, minor = major_minor.split(".", 1)
        return f"0.{major}.{minor}"
    return raw


def update_pyproject(text: str, version: str) -> tuple[str, int]:
    return re.subn(
        r'(^version\s*=\s*")[0-9]+\.[0-9]+\.[0-9]+("\s*$)',
        rf"\g<1>{version}\g<2>",
        text,
        flags=re.MULTILINE,
    )


def update_docs_conf(text: str, version: str) -> tuple[str, int]:
    return re.subn(
        r'(^release\s*=\s*")[0-9]+\.[0-9]+\.[0-9]+("\s*$)',
        rf"\g<1>{version}\g<2>",
        text,
        flags=re.MULTILINE,
    )


def maybe_write(path: Path, content: str, write: bool) -> bool:
    old = path.read_text(encoding="utf-8")
    if old == content:
        return False
    if write:
        path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync Deckard version across docs and metadata",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if files are out of sync",
    )
    mode.add_argument("--write", action="store_true", help="Apply updates in place")
    args = parser.parse_args()

    version = read_changelog_version()
    changed: list[str] = []

    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    pyproject_updated, _ = update_pyproject(pyproject_text, version)
    if maybe_write(PYPROJECT, pyproject_updated, write=args.write):
        changed.append(PYPROJECT.relative_to(ROOT).as_posix())

    docs_text = DOCS_CONF.read_text(encoding="utf-8")
    docs_updated, _ = update_docs_conf(docs_text, version)
    if maybe_write(DOCS_CONF, docs_updated, write=args.write):
        changed.append(DOCS_CONF.relative_to(ROOT).as_posix())

    if changed:
        if args.check:
            print("Deckard version drift detected:")
        else:
            print("Updated Deckard version references:")
        for file in changed:
            print(f"- {file}")
        return 1 if args.check else 0

    print(f"Deckard version synchronized from changelog at {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
