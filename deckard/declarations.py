"""Runtime configuration discovery, parsing, and registration.

This module discovers Hydra YAML configurations from canonical locations and
dynamically registers them with Hydra's ConfigStore at runtime. It supports:

- Built-in canonical config roots (examples/sklearn/config, examples/pytorch/config)
- External config roots via DECKARD_CONFIG_DIRS environment variable
- Optional dependency detection (only register torch configs if torch is available)
- Safe registration via OmegaConf safe_store
- Duplicate registration detection
- Comprehensive logging and debug visibility

Usage:
    >>> from deckard.declarations import register_configs
    >>> register_configs()  # Explicit bootstrap or CLI startup
"""

import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

import yaml
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)


def discover_config_roots() -> List[Path]:
    """
    Discover all active config root directories.

    Returns:
        List[Path]: Ordered list of config root paths to scan. Built-in roots are prioritized before external roots.

    Notes:
        - Built-in roots: examples/sklearn/config, examples/pytorch/config
        - External roots loaded from DECKARD_CONFIG_DIRS environment variable
        - DECKARD_CONFIG_DIRS format: '/path/a:/path/b:/path/c'
        - Only returns directories that exist
    """
    roots: List[Path] = []

    # Discover built-in canonical roots relative to deckard package
    deckard_root = Path(__file__).parent.parent
    builtin_roots = [
        deckard_root / "examples" / "sklearn" / "config",
        deckard_root / "examples" / "pytorch" / "config",
    ]

    for root in builtin_roots:
        if root.is_dir():
            roots.append(root)
            logger.debug(f"Discovered built-in config root: {root}")
        else:
            logger.debug(f"Built-in config root not found: {root}")

    # Discover external config roots from environment variable
    external_dirs = os.environ.get("DECKARD_CONFIG_DIRS", "").strip()
    if external_dirs:
        for path_str in external_dirs.split(":"):
            path_str = path_str.strip()
            if not path_str:
                continue
            path = Path(path_str).expanduser().resolve()
            if path.is_dir():
                roots.append(path)
                logger.debug(f"Discovered external config root: {path}")
            else:
                logger.warning(f"External config root not found: {path}")

    logger.info(f"Discovered {len(roots)} config root(s): {roots}")
    return roots


def iter_config_files(root: Path) -> Iterator[Path]:
    """
    Recursively enumerate YAML configuration files in a root directory.

    Args:
        root (Path): Root directory to scan for .yaml files.

    Yields:
        Path: Paths to .yaml files found in root and subdirectories.

    Notes:
        - Only yields files with .yaml extension
        - Recursively scans subdirectories under the config root
        - Skips hidden directories/files
        - Skips YAML files directly under the config root
        - Skips any file named `default.yaml`
    """
    if not root.is_dir():
        logger.warning(f"Config root is not a directory: {root}")
        return

    for path in sorted(root.rglob("*.yaml")):
        relative_parts = path.relative_to(root).parts

        # Skip hidden files/dirs
        if any(part.startswith(".") for part in relative_parts):
            continue

        # Register only YAML under subfolders of examples/*/config, not config itself.
        if len(relative_parts) < 2:
            logger.debug(f"Skipping root-level config file: {path}")
            continue

        # Never register default.yaml in ConfigStore to avoid schema collisions.
        if path.name == "default.yaml":
            logger.debug(f"Skipping default config registration: {path}")
            continue

        logger.debug(f"Found config file: {path}")
        yield path


def parse_config_file(path: Path) -> Optional[Dict]:
    """
    Parse a YAML configuration file safely.

    Args:
        path (Path): Path to YAML file to parse.

    Returns:
        Optional[Dict]: Parsed YAML as dictionary, or None if parsing fails.

    Notes:
        - Returns None on parse errors (logs warning)
        - Uses safe YAML loader
        - Logs parse errors but continues registration process
    """
    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        if content is None:
            logger.debug(f"Empty config file: {path}")
            return {}
        if not isinstance(content, dict):
            logger.warning(f"Config file does not contain dict root: {path}")
            return None
        return content
    except Exception as e:
        logger.warning(f"Failed to parse config file {path}: {e}")
        return None


def is_package_available(package_name: str) -> bool:
    """
    Check if an optional package is installed.

    Args:
        package_name (str): Name of package to check (e.g., 'torch', 'sklearn').

    Returns:
        bool: True if package is available, False otherwise.

    Notes:
        - Uses importlib.util.find_spec for reliable detection
        - Catches any import exceptions
    """
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _should_register_config(path: Path) -> bool:
    """
    Check if a config should be registered based on package availability.

    Args:
        path (Path): Path to config file.

    Returns:
        bool: True if config should be registered, False if dependencies missing.

    Notes:
        - Pytorch configs require torch to be installed
        - Sklearn configs require sklearn to be installed
        - Other configs are always registered
    """
    path_str = str(path)

    # Check pytorch-specific configs
    if "pytorch" in path_str or "torch" in path_str:
        if not is_package_available("torch"):
            logger.debug(f"Skipping PyTorch config (torch not installed): {path}")
            return False

    # Check sklearn-specific configs
    if "sklearn" in path_str:
        if not is_package_available("sklearn"):
            logger.debug(
                f"Skipping scikit-learn config (sklearn not installed): {path}",
            )
            return False

    return True


def _get_config_group_and_name(path: Path, root: Path) -> tuple:
    """
    Compute Hydra group and config name from file path.

    Args:
        path (Path): Full path to config file.
        root (Path): Root config directory.

    Returns:
        tuple: (group, name) where group is Hydra config group and name is config name.

    Example:
        path: /root/model/sklearn/random_forest.yaml
        root: /root
        returns: ("model/sklearn", "random_forest")

        path: /root/default.yaml
        root: /root
        returns: ("", "default")
    """
    relative_path = path.relative_to(root)
    parts = relative_path.parts[:-1]  # All but filename
    name = relative_path.stem  # Filename without .yaml

    group = "/".join(parts) if parts else ""
    return group, name


def register_configs() -> None:
    """
    Discover, parse, and register all configs with Hydra ConfigStore.

    This is the main entry point for runtime config registration. It:
        1. Discovers all config roots (built-in + external)
        2. Iterates through all YAML files
        3. Checks package availability for conditional registration
        4. Parses YAML files safely
        5. Registers configs with Hydra ConfigStore using safe_store
        6. Logs progress and warnings

    Notes:
        - Safe to call multiple times (logs but doesn't error on duplicates)
        - Skips configs with missing optional dependencies
        - Logs warnings for malformed YAML but continues
        - All errors are logged, never raised
    """
    logger.info("Starting runtime config registration...")

    cs = ConfigStore.instance()
    roots = discover_config_roots()
    registered: Set[str] = set()
    total_files = 0
    registered_count = 0
    skipped_count = 0
    error_count = 0

    for root in roots:
        logger.debug(f"Scanning config root: {root}")

        for config_file in iter_config_files(root):
            total_files += 1

            # Skip configs with missing dependencies
            if not _should_register_config(config_file):
                skipped_count += 1
                continue

            # Parse YAML file
            config_dict = parse_config_file(config_file)
            if config_dict is None:
                error_count += 1
                continue

            # Compute Hydra group and config name
            group, name = _get_config_group_and_name(config_file, root)

            # Detect and log duplicate registrations
            registration_key = f"{group}/{name}" if group else name
            if registration_key in registered:
                logger.warning(
                    f"Duplicate config registration (later registration ignored): {registration_key}",
                )
                continue

            # Register config with Hydra ConfigStore
            try:
                if group:
                    cs.store(name=name, group=group, node=config_dict)
                else:
                    cs.store(name=name, node=config_dict)
                registered.add(registration_key)
                registered_count += 1
                logger.debug(f"Registered config: {registration_key}")
            except Exception as e:
                logger.error(f"Failed to register config {registration_key}: {e}")
                error_count += 1

    logger.info(
        f"Config registration complete: {registered_count} registered, "
        f"{skipped_count} skipped (missing deps), {error_count} errors, "
        f"{total_files} total files",
    )


__all__ = [
    "discover_config_roots",
    "iter_config_files",
    "parse_config_file",
    "is_package_available",
    "register_configs",
]
