import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_optional_dependency_groups() -> set[str]:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return set()

    groups: set[str] = set()
    in_optional_dependencies = False
    for raw_line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            in_optional_dependencies = stripped == "[project.optional-dependencies]"
            continue

        if not in_optional_dependencies or stripped == "" or stripped.startswith("#"):
            continue

        if "=" not in stripped:
            continue

        key = stripped.split("=", 1)[0].strip()
        if key:
            groups.add(key)

    return groups


OPTIONAL_DEP_GROUPS = _load_optional_dependency_groups()


def _extra_from_groups(name: str, aliases: tuple[str, ...] = ()) -> str | None:
    candidates = (name, *aliases)
    for candidate in candidates:
        if candidate in OPTIONAL_DEP_GROUPS:
            return candidate
    return None


PluginRegistry = {
    "anjana": {
        "required_imports": ["anjana", "pycanon"],
        # anjana has explicit dual-import semantics and should not auto-bind to extra groups.
        "extra": None,
    },
    "fairlearn": {
        "required_imports": ["fairlearn"],
        "extra": _extra_from_groups("fairlearn"),
    },
    "lifelines": {
        "required_imports": ["lifelines"],
        "extra": _extra_from_groups("lifelines"),
    },
    "seaborn": {
        "required_imports": ["seaborn"],
        "extra": _extra_from_groups("seaborn"),
    },
    "yellowbrick": {
        "required_imports": ["yellowbrick"],
        "extra": _extra_from_groups("yellowbrick"),
    },
}

FrameworkRegistry = {
    "pytorch": {
        "required_imports": ["torch"],
        "extra": _extra_from_groups("pytorch", aliases=("torch",)),
    },
    "sklearn": {
        "required_imports": ["sklearn"],
        "extra": _extra_from_groups("sklearn"),
    },
    "transformers": {
        "required_imports": ["transformers"],
        "extra": _extra_from_groups("transformers"),
    },
}


def _is_import_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _is_entry_installed(entry: dict[str, Any]) -> bool:
    required = entry.get("required_imports", [])
    return all(_is_import_available(mod) for mod in required)


def _render_summary(summary: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(summary, indent=2, sort_keys=True)

    lines: list[str] = []
    lines.append(f"mode: {summary.get('mode')}")
    lines.append(f"kind: {summary.get('kind')}")
    lines.append(f"requested: {summary.get('requested')}")
    lines.append("items:")
    for item in summary.get("items", []):
        lines.append(
            f"- {item['name']}: installed={item['installed']} action={item['action']} status={item['status']}",
        )
    return "\n".join(lines)


def _run_registry_command(
    *,
    kind: str,
    registry: dict[str, dict[str, Any]],
    name: str | None,
    list_only: bool,
    output_format: str,
) -> dict[str, Any]:
    if name is not None and name not in registry:
        choices = ", ".join(sorted(registry))
        print(f"Unknown {kind[:-1]} '{name}'. Valid options: {choices}", file=sys.stderr)
        raise SystemExit(2)

    target_names = [name] if name else sorted(registry)
    summary: dict[str, Any] = {
        "kind": kind,
        "mode": "list" if list_only else "apply",
        "requested": name if name else "all",
        "items": [],
    }

    for target in target_names:
        entry = registry[target]
        installed = _is_entry_installed(entry)
        item: dict[str, Any] = {
            "name": target,
            "installed": installed,
            "action": "none",
            "status": "listed",
            "extra": entry.get("extra"),
        }

        if not list_only:
            item["action"] = "noop"
            item["status"] = (
                "available-noop" if installed else "optional-dependency-missing"
            )

        summary["items"].append(item)

    print(_render_summary(summary, output_format))

    return summary


def plugins_main(
    plugin: str | None = None,
    list: bool = False,
    format: str = "human",
) -> dict[str, Any]:
    return _run_registry_command(
        kind="plugins",
        registry=PluginRegistry,
        name=plugin,
        list_only=list,
        output_format=format,
    )


def frameworks_main(
    framework: str | None = None,
    list: bool = False,
    format: str = "human",
) -> dict[str, Any]:
    return _run_registry_command(
        kind="frameworks",
        registry=FrameworkRegistry,
        name=framework,
        list_only=list,
        output_format=format,
    )


plugins_parser = argparse.ArgumentParser(add_help=False)
plugins_parser.add_argument("plugin", nargs="?", default=None)
plugins_parser.add_argument("--list", action="store_true", default=False)
plugins_parser.add_argument("--format", choices=["human", "json"], default="human")

frameworks_parser = argparse.ArgumentParser(add_help=False)
frameworks_parser.add_argument("framework", nargs="?", default=None)
frameworks_parser.add_argument("--list", action="store_true", default=False)
frameworks_parser.add_argument("--format", choices=["human", "json"], default="human")
