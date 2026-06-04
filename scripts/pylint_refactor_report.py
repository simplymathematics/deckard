#!/usr/bin/env python3
"""Generate pylint refactor-quality reports for Deckard."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXTRA_RULES = [
    "duplicate-code",
    "too-many-branches",
    "too-many-statements",
    "too-many-locals",
    "too-many-arguments",
    "too-many-nested-blocks",
]


def _load_findings(findings_path: Path) -> list[dict[str, Any]]:
    if not findings_path.exists():
        return []

    raw = findings_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        findings = payload
    elif isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        findings = payload["messages"]
    else:
        findings = []

    return [
        finding
        for finding in findings
        if isinstance(finding, dict)
        and str(finding.get("message-id", "")).startswith("R")
    ]


def _enabled_refactor_rules() -> tuple[list[str], dict[str, str]]:
    enabled_ids: list[str] = []
    enabled_symbols: dict[str, str] = {}
    try:
        output = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pylint",
                "--disable=all",
                "--enable=refactor," + ",".join(EXTRA_RULES),
                "--list-msgs-enabled",
            ],
            text=True,
        )
    except Exception:
        return enabled_ids, enabled_symbols

    pattern = re.compile(r"^([a-z0-9][a-z0-9\-]*?)\s+\((R\d{4})\)$", re.IGNORECASE)
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        symbol, rule_id = match.group(1), match.group(2).upper()
        enabled_ids.append(rule_id)
        enabled_symbols[rule_id] = symbol

    return enabled_ids, enabled_symbols


def _write_markdown_report(
    summary: dict[str, Any],
    findings_path: Path,
    output_md: Path,
    output_json: Path,
) -> None:
    lines = [
        "# Pylint Refactor Quality Report",
        "",
        f"- Total refactor findings: {summary['total_refactor_findings']}",
        f"- Enabled refactor rules: {summary['total_refactor_rules_enabled']}",
        f"- Findings JSON: {findings_path.as_posix()}",
        f"- Summary JSON: {output_json.as_posix()}",
        "",
        "## Refactor Rule Counts",
        "",
        "| Rule | Symbol | Count |",
        "|---|---|---:|",
    ]

    for row in summary["rules"]:
        lines.append(
            f"| {row['message_id']} | {row['symbol'] or '-'} | {row['count']} |",
        )

    lines.extend(
        [
            "",
            "## Top Files by Refactor Findings",
            "",
            "| File | Count |",
            "|---|---:|",
        ],
    )

    top_files = summary["top_files"]
    if top_files:
        for row in top_files:
            lines.append(f"| {row['path']} | {row['count']} |")
    else:
        lines.append("| none | 0 |")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="build/quality_controller",
        help="Directory for report artifacts.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["deckard", "test"],
        help="Pylint targets to scan.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    findings_path = output_dir / "pylint_refactor_findings.json"
    summary_json_path = output_dir / "pylint_refactor_summary.json"
    summary_md_path = output_dir / "pylint_refactor_quality_report.md"

    pylint_cmd = [
        sys.executable,
        "-m",
        "pylint",
        *args.targets,
        "--disable=all",
        "--enable=refactor," + ",".join(EXTRA_RULES),
        "--jobs=0",
        "--output-format=json",
    ]

    result = subprocess.run(
        pylint_cmd,
        text=True,
        capture_output=True,
        check=False,
    )
    findings_path.write_text(result.stdout or "[]", encoding="utf-8")

    findings = _load_findings(findings_path)
    enabled_ids, enabled_symbols = _enabled_refactor_rules()

    counts_by_rule = Counter(
        str(finding.get("message-id", "")) for finding in findings
    )
    counts_by_file = Counter(str(finding.get("path", "")) for finding in findings)
    details_by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for finding in findings:
        rule_id = str(finding.get("message-id", ""))
        details_by_rule[rule_id].append(
            {
                "path": str(finding.get("path", "")),
                "line": int(finding.get("line", 0) or 0),
                "symbol": str(finding.get("symbol", "")),
                "message": str(finding.get("message", "")),
            },
        )

    all_rule_ids = sorted(set(enabled_ids) | set(counts_by_rule.keys()))
    summary = {
        "total_refactor_findings": int(sum(counts_by_rule.values())),
        "total_refactor_rules_enabled": int(len(enabled_ids)),
        "rules": [
            {
                "message_id": rule_id,
                "symbol": enabled_symbols.get(rule_id, ""),
                "count": int(counts_by_rule.get(rule_id, 0)),
            }
            for rule_id in all_rule_ids
        ],
        "top_files": [
            {"path": path, "count": int(count)}
            for path, count in counts_by_file.most_common(20)
        ],
        "details": {
            rule_id: details_by_rule.get(rule_id, []) for rule_id in all_rule_ids
        },
        "pylint_exit_code": int(result.returncode),
    }

    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_report(summary, findings_path, summary_md_path, summary_json_path)

    print(f"TOTAL_REFACTOR_FINDINGS {summary['total_refactor_findings']}")
    print(f"ENABLED_REFACTOR_RULES {summary['total_refactor_rules_enabled']}")
    print(f"PYLINT_EXIT_CODE {summary['pylint_exit_code']}")
    print(f"REPORT_MD {summary_md_path.as_posix()}")
    print(f"REPORT_JSON {summary_json_path.as_posix()}")


if __name__ == "__main__":
    main()
