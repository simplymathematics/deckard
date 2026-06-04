#!/usr/bin/env python3
"""Enforce per-file and per-group Python coverage thresholds from text reports.

This script parses coverage text output (for example build/coverage.txt) and
enforces:
- per-file threshold for selected prefixes
- aggregate threshold for core modules
- aggregate threshold for each framework under deckard/frameworks
- aggregate threshold for each plugin under deckard/plugins
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = "build/coverage.txt"
DEFAULT_THRESHOLD = 80.0

LINE_RE = re.compile(
    r"^(?P<path>\S+)\s+\d+\s+\d+\s+(?P<cover>\d+)%",
)

CORE_MODULES = (
    "attack",
    "data",
    "detector",
    "experiment",
    "layers",
    "model",
    "plot",
    "score",
)


@dataclass(frozen=True)
class CoverageEntry:
    path: str
    cover: float


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return ROOT / path


def _parse_report(report_path: Path) -> list[CoverageEntry]:
    entries: list[CoverageEntry] = []
    for raw_line in report_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("-") or line.startswith("TOTAL"):
            continue
        match = LINE_RE.match(line)
        if match is None:
            continue
        path = match.group("path")
        if not path.startswith("deckard/"):
            continue
        entries.append(CoverageEntry(path=path, cover=float(match.group("cover"))))
    return entries


def _mean(values: list[float]) -> float:
    if not values:
        return 100.0
    return sum(values) / len(values)


def _prefix_entries(entries: list[CoverageEntry], prefix: str) -> list[CoverageEntry]:
    normalized = prefix.rstrip("/") + "/"
    return [entry for entry in entries if entry.path.startswith(normalized)]


def _collect_framework_groups(
    entries: list[CoverageEntry],
) -> dict[str, list[CoverageEntry]]:
    groups: dict[str, list[CoverageEntry]] = {}
    for entry in entries:
        if not entry.path.startswith("deckard/frameworks/"):
            continue
        parts = entry.path.split("/")
        if len(parts) < 4:
            continue
        if parts[2].endswith(".py"):
            continue
        group = f"deckard/frameworks/{parts[2]}"
        groups.setdefault(group, []).append(entry)
    return groups


def _collect_plugin_groups(
    entries: list[CoverageEntry],
) -> dict[str, list[CoverageEntry]]:
    groups: dict[str, list[CoverageEntry]] = {}
    for entry in entries:
        if not entry.path.startswith("deckard/plugins/"):
            continue
        parts = entry.path.split("/")
        if len(parts) < 4:
            continue
        if parts[2].endswith(".py"):
            continue
        group = f"deckard/plugins/{parts[2]}"
        groups.setdefault(group, []).append(entry)
    return groups


def _collect_core_groups(
    entries: list[CoverageEntry],
) -> dict[str, list[CoverageEntry]]:
    groups: dict[str, list[CoverageEntry]] = {}
    for module in CORE_MODULES:
        prefix = f"deckard/{module}/"
        module_entries = [entry for entry in entries if entry.path.startswith(prefix)]
        if module_entries:
            groups[f"deckard/{module}"] = module_entries
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce Python coverage thresholds by file and module group",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help="Coverage report text file (default: build/coverage.txt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum coverage percentage for file/group checks (default: 80)",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help=(
            "Coverage prefixes to enforce per-file. "
            "Repeatable, e.g. --prefix deckard/data --prefix deckard/model"
        ),
    )
    parser.add_argument(
        "--enforce-core-modules",
        action="store_true",
        help="Enforce aggregate threshold for each core module under deckard/",
    )
    parser.add_argument(
        "--enforce-frameworks",
        action="store_true",
        help="Enforce aggregate threshold for each deckard/frameworks/<name>",
    )
    parser.add_argument(
        "--enforce-plugins",
        action="store_true",
        help="Enforce aggregate threshold for each deckard/plugins/<name>",
    )
    args = parser.parse_args()

    report_path = _resolve_path(args.report)
    entries = _parse_report(report_path)
    if not entries:
        print(f"No deckard coverage entries found in {report_path}")
        return 2

    violations: list[str] = []
    threshold = float(args.threshold)

    prefixes = list(args.prefix)
    if not prefixes:
        prefixes = ["deckard/data"]

    for prefix in prefixes:
        scoped = _prefix_entries(entries, prefix)
        if not scoped:
            violations.append(f"No coverage entries found for prefix: {prefix}")
            continue
        for entry in scoped:
            if entry.cover < threshold:
                violations.append(
                    f"FILE {entry.path}: {entry.cover:.1f}% < {threshold:.1f}%",
                )

    grouped: dict[str, list[CoverageEntry]] = {}
    if args.enforce_core_modules:
        grouped.update(_collect_core_groups(entries))
    if args.enforce_frameworks:
        grouped.update(_collect_framework_groups(entries))
    if args.enforce_plugins:
        grouped.update(_collect_plugin_groups(entries))

    for group_name, group_entries in sorted(grouped.items()):
        group_cover = _mean([entry.cover for entry in group_entries])
        if group_cover < threshold:
            violations.append(
                f"GROUP {group_name}: {group_cover:.1f}% < {threshold:.1f}%",
            )

    print(f"Checked {len(entries)} deckard coverage entries from {report_path}")
    print(f"Threshold: {threshold:.1f}%")
    if violations:
        print("Coverage gate failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("Coverage gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
