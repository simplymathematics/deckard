#!/usr/bin/env python3
"""Compute and enforce docs coverage quality from docs enforcement audit data.

Metric definition:
- docs_coverage_percent = 100 * clean_files / files_checked
- clean_file = docs file with zero docs enforcement violations

The script can either:
1) collect audit data directly from docs scopes, or
2) read a precomputed audit JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repository_enforcement import collect_docs_audit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_JSON = "build/quality_controller/docs_coverage_audit.json"
DEFAULT_SUMMARY_MD = "build/quality_controller/docs_coverage_summary.md"


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _parse_docs_scopes(raw_scope: str) -> list[str]:
    value = str(raw_scope).strip()
    if not value:
        return ["docs"]
    return [part.strip() for part in value.split(",") if part.strip()]


def _extract_entries(payload: Any) -> list[dict[str, Any]]:
    if (
        isinstance(payload, dict)
        and "files" in payload
        and isinstance(payload["files"], list)
    ):
        entries = payload["files"]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(
            "Audit JSON must be either a list of entries or a payload with a 'files' list.",
        )

    normalized: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "path": str(item.get("path", "")),
                "lines_checked": int(item.get("lines_checked", 0) or 0),
                "violation_count": int(item.get("violation_count", 0) or 0),
                "violation_codes": dict(item.get("violation_codes", {}) or {}),
            },
        )
    return normalized


def _compute_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    files_checked = len(entries)
    files_with_violations = sum(1 for e in entries if e["violation_count"] > 0)
    clean_files = files_checked - files_with_violations
    total_violations = sum(int(e["violation_count"]) for e in entries)
    total_lines_checked = sum(int(e["lines_checked"]) for e in entries)

    docs_coverage_percent = (
        (100.0 * clean_files / files_checked) if files_checked > 0 else 100.0
    )

    line_quality_percent = (
        (100.0 * (1.0 - (total_violations / total_lines_checked)))
        if total_lines_checked > 0
        else 100.0
    )

    return {
        "files_checked": files_checked,
        "clean_files": clean_files,
        "files_with_violations": files_with_violations,
        "total_violations": total_violations,
        "total_lines_checked": total_lines_checked,
        "docs_coverage_percent": docs_coverage_percent,
        "line_quality_percent": line_quality_percent,
    }


def _write_json_report(output_json: Path, payload: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_markdown_report(
    output_md: Path,
    summary: dict[str, Any],
    threshold: float,
    top_offenders: list[dict[str, Any]],
    docs_scopes: list[str],
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if summary["docs_coverage_percent"] >= threshold else "FAIL"

    lines = [
        "# Docs Coverage Quality Gate",
        "",
        f"- Status: **{status}**",
        f"- Docs scopes: {', '.join(docs_scopes)}",
        f"- Threshold: {threshold:.2f}%",
        f"- Docs coverage: {summary['docs_coverage_percent']:.2f}%",
        f"- Line quality: {summary['line_quality_percent']:.2f}%",
        f"- Files checked: {summary['files_checked']}",
        f"- Clean files: {summary['clean_files']}",
        f"- Files with violations: {summary['files_with_violations']}",
        f"- Total violations: {summary['total_violations']}",
        "",
    ]

    if top_offenders:
        lines.extend(
            [
                "## Top offenders",
                "",
                "| File | Violations |",
                "|---|---:|",
            ],
        )
        for item in top_offenders:
            lines.append(f"| {item['path']} | {item['violation_count']} |")
        lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce docs coverage quality gate")
    parser.add_argument(
        "--docs-scope",
        default="docs",
        help="Docs scope(s), comma-separated when collecting live audit data.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=95.0,
        help="Minimum docs coverage percent required to pass (default: 95.0).",
    )
    parser.add_argument(
        "--audit-json-input",
        default="",
        help="Optional precomputed docs audit JSON input. If omitted, audit is collected live.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_AUDIT_JSON,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--output-md",
        default=DEFAULT_SUMMARY_MD,
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top offending files to include in markdown output.",
    )
    args = parser.parse_args()

    docs_scopes = _parse_docs_scopes(args.docs_scope)

    if args.audit_json_input:
        payload = json.loads(
            _resolve_path(args.audit_json_input).read_text(encoding="utf-8"),
        )
        entries = _extract_entries(payload)
    else:
        entries = []
        for scope in docs_scopes:
            entries.extend(collect_docs_audit(scope))

    entries = sorted(entries, key=lambda e: str(e.get("path", "")))
    summary = _compute_summary(entries)

    top_offenders = sorted(
        [e for e in entries if int(e["violation_count"]) > 0],
        key=lambda e: int(e["violation_count"]),
        reverse=True,
    )[: max(args.top, 0)]

    report_payload = {
        "docs_scopes": docs_scopes,
        "threshold": float(args.threshold),
        "summary": summary,
        "files": entries,
    }

    output_json = _resolve_path(args.output_json)
    output_md = _resolve_path(args.output_md)
    _write_json_report(output_json, report_payload)
    _write_markdown_report(
        output_md,
        summary,
        float(args.threshold),
        top_offenders,
        docs_scopes,
    )

    print(
        f"Docs coverage: {summary['docs_coverage_percent']:.2f}% (threshold: {args.threshold:.2f}%)",
    )
    print(
        f"Files checked: {summary['files_checked']}, clean files: {summary['clean_files']}",
    )
    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote markdown report: {output_md}")

    if summary["docs_coverage_percent"] < args.threshold:
        print("Docs coverage gate failed.")
        return 1

    print("Docs coverage gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
