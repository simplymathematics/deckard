"""Shared OmegaConf resolver registration for deckard configs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


def _load_yaml_file(path: Path):
    """Load a YAML file from disk and return the parsed Python object."""
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def _resolve_config_dir() -> str:
    """Return the active deckard config directory.

    The historical contract allows tests and callers to monkeypatch
    ``deckard.DECKARD_CONFIG_DIR`` directly, so we honor that module-level
    override before falling back to the environment variable.
    """
    try:
        module_names = (
            "deckard",
            "deckard.experiment.base",
        )
        for module_name in module_names:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            module_dir = getattr(module, "DECKARD_CONFIG_DIR", None)
            if module_dir and str(module_dir) != "config":
                return str(module_dir)
    except Exception:
        pass
    return os.environ.get("DECKARD_CONFIG_DIR", "config")


def _file_resolver(arg: str):
    """Resolve ``${file:...}`` OmegaConf interpolations relative to deckard config."""
    if not arg:
        raise ValueError(
            "file resolver requires an argument like 'path/to/file.yaml[:key]'",
        )

    if ":" in arg:
        path_part, key_part = arg.split(":", 1)
        key_part = key_part.strip()
    else:
        path_part, key_part = arg, None

    config_dir = _resolve_config_dir()
    path = Path(config_dir, path_part)
    if not path.exists():
        raise FileNotFoundError(
            f"file resolver: file not found: {path_part} in working dir {os.getcwd()}",
        )

    data = _load_yaml_file(path)
    if key_part:
        parts = key_part.split(".")
        cur: Any = data
        for part in parts:
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                raise KeyError(
                    f"file resolver: key '{key_part}' not found in {path}",
                )
        data = cur
    return OmegaConf.create(data)


def _merge_resolver(*args):
    """Resolve and merge multiple config fragments into a single OmegaConf node."""
    merged = OmegaConf.create()
    for arg in args:
        obj = OmegaConf.to_container(OmegaConf.create(arg), resolve=True)
        merged = OmegaConf.merge(merged, obj)
    return OmegaConf.create(merged)


def register_core_resolvers() -> None:
    """Register deckard's core OmegaConf resolvers.

    The registration is idempotent because each resolver is installed with
    ``replace=True``.
    """
    OmegaConf.register_new_resolver(
        "file",
        _file_resolver,
        replace=True,
        use_cache=True,
    )
    OmegaConf.register_new_resolver("merge", _merge_resolver, replace=True)
