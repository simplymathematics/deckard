from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"


def _iter_toctree_targets(text: str) -> list[str]:
    targets: list[str] = []
    lines = text.splitlines()
    in_toctree = False
    for line in lines:
        stripped = line.strip()
        if stripped == "```{toctree}":
            in_toctree = True
            continue
        if in_toctree and stripped == "```":
            in_toctree = False
            continue
        if not in_toctree:
            continue
        if not stripped or stripped.startswith(":"):
            continue
        targets.append(stripped)
    return targets


def _candidate_doc_targets(source: Path, target: str) -> list[Path]:
    if target.startswith("/"):
        resolved = DOCS_ROOT / target.lstrip("/")
    else:
        resolved = (source.parent / target).resolve()
    if resolved.suffix:
        return [resolved]
    return [
        resolved.with_suffix(".md"),
        resolved.with_suffix(".ipynb"),
        resolved / "index.md",
    ]


def _iter_markdown_targets(text: str) -> list[str]:
    matches = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    return [match for match in matches if not match.startswith(("http://", "https://", "#"))]


def _iter_doc_role_targets(text: str) -> list[str]:
    return re.findall(r"\{doc\}`[^`<]*<([^>]+)>`", text)


def _assert_targets_exist(source: Path) -> None:
    text = source.read_text(encoding="utf-8")
    missing: list[str] = []

    for target in _iter_toctree_targets(text):
        candidates = _candidate_doc_targets(source, target)
        if not any(candidate.exists() for candidate in candidates):
            missing.append(target)

    for target in _iter_markdown_targets(text):
        candidates = _candidate_doc_targets(source, target)
        if not any(candidate.exists() for candidate in candidates):
            missing.append(target)

    for target in _iter_doc_role_targets(text):
        candidates = _candidate_doc_targets(source, target)
        if not any(candidate.exists() for candidate in candidates):
            missing.append(target)

    assert not missing, f"Broken documentation targets in {source}: {sorted(set(missing))}"


def test_notebook_index_references_resolve() -> None:
    _assert_targets_exist(DOCS_ROOT / "notebooks" / "index.md")


def test_overview_index_references_resolve() -> None:
    _assert_targets_exist(DOCS_ROOT / "overview" / "index.md")