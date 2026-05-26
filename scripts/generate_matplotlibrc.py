#!/usr/bin/env python3
"""Generate a matplotlibrc file from seaborn theme settings.

This script applies ``seaborn.set_theme`` with user-provided parameters, then
writes the resulting ``matplotlib.rcParams`` to a matplotlibrc-style file.

Supported set_theme parameters:
- context
- style
- palette
- font
- font_scale
- color_codes
- rc
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import seaborn as sns


def _parse_json_or_literal(value: str) -> Any:
    """Parse value as JSON first, then Python literal, else return raw string."""
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return value


def _parse_maybe_mapping(value: str | None) -> str | dict[str, Any] | None:
    """Parse a string into a dict when possible, otherwise keep it as string."""
    if value is None:
        return None
    parsed = _parse_json_or_literal(value)
    if isinstance(parsed, dict):
        return parsed
    return str(parsed)


def _parse_palette(value: str | None) -> str | list[Any] | tuple[Any, ...] | None:
    """Parse palette as sequence when provided as literal, else keep as string."""
    if value is None:
        return None
    parsed = _parse_json_or_literal(value)
    if isinstance(parsed, (list, tuple, str)):
        return parsed
    return str(parsed)


def _parse_rc_overrides(
    rc_value: str | None,
    rc_params: list[str] | None,
) -> dict[str, Any] | None:
    """Parse rc mappings from --rc and --rc-param values."""
    merged: dict[str, Any] = {}

    if rc_value is not None:
        parsed = _parse_json_or_literal(rc_value)
        if not isinstance(parsed, dict):
            raise ValueError("--rc must parse to a dictionary mapping")
        merged.update(parsed)

    for item in rc_params or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --rc-param '{item}'. Expected format key=value",
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --rc-param '{item}'. Key cannot be empty")
        merged[key] = _parse_json_or_literal(raw_value)

    return merged or None


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate a matplotlibrc file from seaborn set_theme parameters.",
    )
    parser.add_argument(
        "--context",
        default="notebook",
        help=(
            "Seaborn context (string) or mapping literal/JSON; "
            "e.g. notebook, talk, or '{\"axes.titlesize\": 14}'"
        ),
    )
    parser.add_argument(
        "--style",
        default="whitegrid",
        help=(
            "Seaborn style (string) or mapping literal/JSON; "
            "e.g. whitegrid, dark, ticks"
        ),
    )
    parser.add_argument(
        "--palette",
        default="colorblind",
        help=(
            "Palette name or sequence literal/JSON; "
            'e.g. colorblind or \'["#4c72b0", "#dd8452"]\''
        ),
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Font family string (e.g. 'DejaVu Sans').",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Separate scaling factor for font elements.",
    )
    parser.add_argument(
        "--color-codes",
        dest="color_codes",
        action="store_true",
        default=True,
        help="Enable seaborn shorthand color code remapping (default: enabled).",
    )
    parser.add_argument(
        "--no-color-codes",
        dest="color_codes",
        action="store_false",
        help="Disable seaborn shorthand color code remapping.",
    )
    parser.add_argument(
        "--rc",
        default=None,
        help=(
            "rc override mapping as JSON or Python-literal dict; "
            "e.g. '{\"axes.spines.top\": false}'"
        ),
    )
    parser.add_argument(
        "--rc-param",
        action="append",
        default=None,
        help=(
            "Single rc override in key=value format; can be passed multiple times. "
            "Values are parsed as JSON/Python literals when possible."
        ),
    )
    parser.add_argument(
        "--output",
        default="deckard/plot/.matplotlibrc",
        help="Output file path for matplotlibrc content (default: deckard/plot/.matplotlibrc).",
    )
    return parser


def write_matplotlibrc(output_path: Path) -> None:
    """Write current matplotlib rcParams to output path."""

    def _format_rc_value(value: Any) -> str | None:
        # Matplotlibrc expects comma-separated values without list brackets.
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        return str(value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for key, value in mpl.rcParams.items():
            serialized = _format_rc_value(value)
            if serialized is None:
                continue
            handle.write(f"{key}: {serialized}\n")


def main() -> int:
    """Run CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    context = _parse_maybe_mapping(args.context)
    style = _parse_maybe_mapping(args.style)
    palette = _parse_palette(args.palette)
    rc = _parse_rc_overrides(args.rc, args.rc_param)

    theme_kwargs: dict[str, Any] = {
        "context": context,
        "style": style,
        "palette": palette,
        "font_scale": args.font_scale,
        "color_codes": args.color_codes,
        "rc": rc,
    }
    if args.font is not None:
        theme_kwargs["font"] = args.font

    sns.set_theme(
        **theme_kwargs,
    )

    output_path = Path(args.output).expanduser().resolve()
    write_matplotlibrc(output_path)
    print(f"Wrote matplotlibrc to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
