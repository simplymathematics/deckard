#!/usr/bin/env python

import argparse
import inspect
import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from . import DECKARD_CONFIG_DIR
from .declarations import register_configs
from .layers import SUPPORTED_LAYERS, layer_dict
from .utils import normalize_hydra_list_overrides

# Set up logging
logger = logging.getLogger(__name__)


def _layer_help_summary(layer_name: str) -> str:
    """Return a concise layer help summary from the registered layer parser."""
    parser = layer_dict[layer_name][0]
    description = getattr(parser, "description", None)
    if isinstance(description, str):
        summary = description.strip()
        if summary:
            return summary
    usage = getattr(parser, "usage", None)
    if isinstance(usage, str):
        usage = usage.strip()
        if usage:
            return f"Run layer ({usage})"
    return f"Run the {layer_name} layer"


def _forward_hydra_control_args(parsed_args) -> list[str]:
    """
    Rebuild Hydra control CLI flags parsed by Hydra's own parser.

    Args:
        parsed_args (argparse.Namespace): Parsed arguments from Hydra's parser.

    Returns:
        list[str]: List of CLI flags to forward to Hydra.

    Example:
        >>> _forward_hydra_control_args(parsed_args)
        ['--multirun', '--cfg', 'config.yaml']
    """
    forwarded: list[str] = []

    bool_flags = {
        "run": "--run",
        "multirun": "--multirun",
        "shell_completion": "--shell-completion",
        "hydra_help": "--hydra-help",
        "help": "--help",
        "resolve": "--resolve",
    }
    for attr, cli_flag in bool_flags.items():
        if bool(getattr(parsed_args, attr, False)):
            forwarded.append(cli_flag)

    value_flags = {
        "cfg": "--cfg",
        "package": "--package",
        "info": "--info",
        "experimental_rerun": "--experimental-rerun",
    }
    for attr, cli_flag in value_flags.items():
        value = getattr(parsed_args, attr, None)
        if value not in [None, ""]:
            forwarded.extend([cli_flag, str(value)])

    return forwarded


def get_configuration_paths():
    """
    Discover and validate the configuration directory and file paths.

    Returns:
        tuple[str, str]: Tuple of (config_dir, config_file) as absolute paths.

    Raises:
        FileNotFoundError: If the config file does not exist in the specified directory.
    """
    # Get config dir from environment variable if set
    config_dir = os.environ.get(
        "DECKARD_CONFIG_DIR",
        DECKARD_CONFIG_DIR,
    )
    if config_dir is None:
        # TODO: Read from existing .deckard_rc/create one
        logger.error(
            "DECKARD_CONFIG_DIR must be specified as an environment variable.",
        )
        sys.exit(1)

    logger.debug("No optional arguments provided.")
    requested_config_file = Path(
        os.environ.get("DECKARD_DEFAULT_CONFIG_FILE") or "default.yaml",
    ).as_posix()
    config_file = requested_config_file
    working_dir = os.getcwd()
    logger.info(f"Current working directory: {working_dir}")
    logger.info("Starting deckard with Hydra configuration.")
    logger.info(f"Config directory: {Path(config_dir).resolve()}")
    while not Path(config_dir).exists():
        config_dir = input(
            f"Config directory '{config_dir}' does not exist. "
            "Please enter a valid config directory path: ",
        )
    if not Path(config_dir).is_absolute():
        config_dir = os.path.relpath(config_dir, working_dir)
    requested_config_path = Path(config_dir, requested_config_file)
    if not requested_config_path.exists() and requested_config_file != "default.yaml":
        fallback_config_path = Path(config_dir, "default.yaml")
        if fallback_config_path.exists():
            logger.warning(
                "Configured default file '%s' not found in '%s'; falling back to 'default.yaml'.",
                requested_config_file,
                config_dir,
            )
            config_file = "default.yaml"

    logger.info(f"Resolved config file path: {config_file}")
    if not Path(config_dir, config_file).exists():
        logger.error(
            f"Config file {config_file} does not exist. Did you set DECKARD_CONFIG_DIR correctly?",
        )
        raise FileNotFoundError(config_file)
    return config_dir, config_file


def _build_router() -> argparse.ArgumentParser:
    """
    Build a minimal routing parser for deckard CLI subcommands.

    Returns:
        argparse.ArgumentParser: Argument parser with subcommands for each supported layer.
    """
    parser = argparse.ArgumentParser(
        prog="deckard",
        description=(
            "deckard command-line interface. "
            "Select a layer module and pass remaining arguments through to that layer parser and Hydra."
        ),
    )
    subs = parser.add_subparsers(dest="module", metavar="MODULE", required=True)
    for name in layer_dict:
        sub = subs.add_parser(
            name,
            help=_layer_help_summary(name),
            description=_layer_help_summary(name),
            add_help=False,
        )
        sub.add_argument(
            "remainder",
            nargs=argparse.REMAINDER,
            help=(
                "Arguments forwarded to the selected layer parser/Hydra "
                "(for example: --config-dir, --config-name, overrides)."
            ),
        )
    return parser


def main():
    """
    Main entry point for the deckard CLI.

    Handles config directory setup, runtime config registration, and dispatches
    to the appropriate layer entrypoint based on CLI arguments.
    """
    parser = _build_router()

    config_dir = os.environ.get("DECKARD_CONFIG_DIR", "config")
    while not Path(config_dir).exists():
        config_dir = input(
            f"Config directory '{config_dir}' does not exist. "
            "Please enter a valid config directory path: ",
        )
    os.environ["DECKARD_CONFIG_DIR"] = Path(config_dir).resolve().as_posix()

    if os.environ.get("DECKARD_SKIP_RUNTIME_CONFIG_REGISTRATION", "0") not in {
        "1",
        "true",
        "True",
        "yes",
        "on",
    }:
        register_configs()

    parsed, _ = parser.parse_known_args()
    module = parsed.module

    sys.argv.pop(sys.argv.index(module))
    if module in SUPPORTED_LAYERS:
        generate_hydra_main(module)
    else:
        raise ValueError(
            f"Module: {module} not supported. Must be one of {SUPPORTED_LAYERS}",
        )


def generate_hydra_main(layer):
    """
    Run the parser and main entrypoint for the specified layer via Hydra.

    Args:
        layer (str): Name of the layer to execute.

    Raises:
        ValueError: If the specified layer is not supported.
    """
    if layer not in layer_dict:
        logger.error(
            f"Unsupported layer: {layer}. Supported layers are: {list(layer_dict)}",
        )
        raise ValueError

    parser, main_fn = layer_dict[layer]
    if not hasattr(parser, "parse_known_args"):
        raise ValueError("Parser object does not have .parse_known_args")

    # Parse layer-specific args first, then leave remaining args for Hydra.
    parsed_args, hydra_args = parser.parse_known_args(sys.argv[1:])

    cli_config_path = getattr(parsed_args, "config_path", None) or getattr(
        parsed_args,
        "config_dir",
        None,
    )
    cli_config_name = getattr(parsed_args, "config_name", None)
    default_config_dir, default_config_file = get_configuration_paths()
    config_dir = cli_config_path if cli_config_path else default_config_dir
    config_file = cli_config_name if cli_config_name else default_config_file

    forwarded_overrides = []
    if hasattr(parsed_args, "overrides") and isinstance(
        parsed_args.overrides,
        list,
    ):
        # get_args_parser may parse Hydra key=value arguments into `overrides`.
        forwarded_overrides = normalize_hydra_list_overrides(
            parsed_args.overrides,
            keys=("score",),
        )
    forwarded_control_args = _forward_hydra_control_args(parsed_args)
    sys.argv = [
        sys.argv[0],
        *forwarded_control_args,
        *hydra_args,
        *forwarded_overrides,
    ]

    @hydra.main(
        config_path=(
            str(Path(config_dir).resolve()) if config_dir is not None else None
        ),
        config_name=config_file,
        version_base="1.3",
    )
    def main_hydra(cfg: DictConfig) -> None:
        raw_args = vars(parsed_args).copy()
        sig = inspect.signature(main_fn)
        valid_keys = set(sig.parameters.keys())
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        args = (
            raw_args.copy()
            if accepts_var_kwargs
            else {k: v for k, v in raw_args.items() if k in valid_keys}
        )

        if "cfg" in valid_keys:
            args["cfg"] = cfg

        # Allow Hydra overrides for parser keys when present.
        if cfg is not None:
            for key in list(args.keys()):
                if key in cfg and cfg[key] is not None:
                    args[key] = cfg[key]
        return main_fn(**args)

    return main_hydra()


if __name__ == "__main__":
    main()
