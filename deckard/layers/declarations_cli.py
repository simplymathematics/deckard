import argparse
import json
import sys
from typing import Any

import yaml

from deckard import declarations as decl


def _parse_selector(selector: str) -> list[str]:
    return [part for part in selector.strip("/").split("/") if part]


def _require_full_selector(selector: str) -> tuple[str, str, str]:
    parts = _parse_selector(selector)
    if len(parts) < 3:
        print(
            "Selector must be '<component>/<subcomponent>/<name>'.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return parts[0], parts[1], "/".join(parts[2:])


def _render(data: Any, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(data, indent=2, sort_keys=True)
    if output_format == "yaml":
        return yaml.safe_dump(data, sort_keys=False)

    # tree format
    if isinstance(data, dict) and "items" in data:
        lines = [
            f"kind: {data.get('kind')}",
            f"requested: {data.get('requested')}",
            "items:",
        ]
        for item in data.get("items", []):
            lines.append(f"- {item}")
        return "\n".join(lines)

    return yaml.safe_dump(data, sort_keys=False)


def _normalize_root(value: str | None) -> str | None:
    if value in {None, "", "all"}:
        return None
    return value


def _resolve_selector(
    *,
    selector: str,
    root_kind: str | None,
    index: list[decl.DeclarationIndexEntry],
) -> decl.DeclarationIndexEntry:
    entry = decl.get_declaration_by_selector(
        selector,
        root_kind=root_kind,
        index=index,
    )
    if entry is None:
        print(f"Declaration '{selector}' was not found.", file=sys.stderr)
        raise SystemExit(2)
    return entry


def declarations_main(
    command: str,
    selector: str | None = None,
    root: str | None = None,
    format: str = "tree",
    resolve: bool = False,
    strict: bool = False,
    set: list[str] | None = None,
) -> dict[str, Any]:
    root_kind = _normalize_root(root)
    index = decl.discover_declaration_index()
    if root_kind is not None:
        index = [entry for entry in index if entry.root_kind == root_kind]

    summary: dict[str, Any] = {
        "command": command,
        "requested": selector,
        "root": root or "all",
    }

    if command == "list":
        if selector is None:
            items = sorted({entry.component for entry in index})
            summary["kind"] = "component"
            summary["items"] = items
        else:
            parts = _parse_selector(selector)
            if len(parts) == 1:
                component = parts[0]
                items = sorted(
                    {
                        entry.subcomponent
                        for entry in index
                        if entry.component == component
                    },
                )
                summary["kind"] = "subcomponent"
                summary["items"] = items
            elif len(parts) == 2:
                component, subcomponent = parts
                items = sorted(
                    {
                        entry.selector
                        for entry in index
                        if entry.component == component
                        and entry.subcomponent == subcomponent
                    },
                )
                summary["kind"] = "declaration"
                summary["items"] = items
            else:
                print(
                    "List accepts '<component>' or '<component>/<subcomponent>'.",
                    file=sys.stderr,
                )
                raise SystemExit(2)

    elif command == "show":
        component, subcomponent, name = _require_full_selector(selector or "")
        entry = _resolve_selector(
            selector=f"{component}/{subcomponent}/{name}",
            root_kind=root_kind,
            index=index,
        )
        summary["entry"] = {
            "selector": entry.selector,
            "group": entry.group,
            "name": entry.name,
            "file": str(entry.path),
            "root": entry.root_kind,
            "payload": decl.load_declaration_payload(entry),
        }

    elif command == "validate":
        component, subcomponent, name = _require_full_selector(selector or "")
        entry = _resolve_selector(
            selector=f"{component}/{subcomponent}/{name}",
            root_kind=root_kind,
            index=index,
        )
        validation = decl.validate_declaration(entry)
        if strict and validation.get("warnings"):
            validation["valid"] = False
            validation["error"] = "Strict validation failed due to warnings."
        summary["result"] = validation

    elif command == "compose":
        component, subcomponent, name = _require_full_selector(selector or "")
        full_selector = f"{component}/{subcomponent}/{name}"
        entry = _resolve_selector(
            selector=full_selector,
            root_kind=root_kind,
            index=index,
        )
        payload = decl.compose_declaration(entry, overrides=set or [])
        summary["result"] = {
            "selector": full_selector,
            "resolved": bool(resolve),
            "payload": payload,
        }

    else:
        print(f"Unsupported command '{command}'.", file=sys.stderr)
        raise SystemExit(2)

    print(_render(summary, format))
    return summary


declarations_parser = argparse.ArgumentParser(add_help=False)
declarations_parser.add_argument(
    "command",
    choices=["list", "show", "validate", "compose"],
)
declarations_parser.add_argument("selector", nargs="?", default=None)
declarations_parser.add_argument(
    "--root",
    choices=["all", "sklearn", "pytorch", "external"],
    default="all",
)
declarations_parser.add_argument(
    "--format",
    choices=["tree", "json", "yaml"],
    default="tree",
)
declarations_parser.add_argument("--resolve", action="store_true", default=False)
declarations_parser.add_argument("--strict", action="store_true", default=False)
declarations_parser.add_argument("--set", action="append", default=[])
