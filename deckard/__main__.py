#!/usr/bin/env python

import argparse
import logging
import os
import sys
import inspect
import hydra
from omegaconf import DictConfig
from pathlib import Path


from . import DECKARD_CONFIG_DIR, DECKARD_DEFAULT_CONFIG_FILE

from .layers import SUPPORTED_LAYERS, layer_dict
from .experiment import ExperimentConfig

# Set up logging
logger = logging.getLogger(__name__)


def _forward_hydra_control_args(parsed_args) -> list[str]:
    """Rebuild Hydra control CLI flags parsed by Hydra's own parser.

    `get_args_parser()` consumes flags like `--multirun` and `--cfg` into
    parsed_args fields. When we later rebuild `sys.argv` for `@hydra.main`,
    these flags must be forwarded explicitly or Hydra falls back to single-run
    mode.
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
    """ """
    # Get config dir from environment variable if set
    config_dir = os.environ.get(
        "DECKARD_CONFIG_DIR",
        DECKARD_CONFIG_DIR,
    )
    if config_dir is None:
        logger.error("DECKARD_CONFIG_DIR must be specified as an environment variable.")
        sys.exit(1)
    while not Path(config_dir).exists():
        # Deckard_config dir does not exist, have the user set it using input()
        config_dir = input(
            f"The provided config directory path '{config_dir}' does not exist. Please enter a valid config directory path: ",
        )
        # Prompt user to confirm the path exists
        if not Path(config_dir).exists():
            config_dir = None
    logger.debug("No optional arguments provided.")
    config_file = Path(
        os.environ.get("DECKARD_DEFAULT_CONFIG_FILE", DECKARD_DEFAULT_CONFIG_FILE),
    ).as_posix()
    working_dir = os.getcwd()
    logger.info(f"Current working directory: {working_dir}")
    logger.info("Starting Deckard with Hydra configuration.")
    logger.info(f"Config directory: {Path(config_dir).resolve()}")
    if not Path(config_dir).is_absolute():
        config_dir = os.path.relpath(config_dir, working_dir)
    logger.info(f"Resolved config file path: {config_file}")
    if not Path(config_dir, config_file).exists():
        logger.error(
            f"Config file {config_file} does not exist. Did you set DECKARD_CONFIG_DIR correctly?",
        )
        raise FileNotFoundError(config_file)
    return config_dir, config_file


def _build_router() -> argparse.ArgumentParser:
    """Minimal routing parser: recognises the subcommand name and passes everything else through."""
    parser = argparse.ArgumentParser(prog="deckard", description="Deckard command-line interface")
    subs = parser.add_subparsers(dest="module", metavar="MODULE", required=True)
    for name in layer_dict:
        sub = subs.add_parser(name, help=f"Run the {name} layer", add_help=False)
        sub.add_argument("remainder", nargs=argparse.REMAINDER)
    return parser


def main():
    parser = _build_router()

    config_dir = os.environ.get("DECKARD_CONFIG_DIR", "config")
    while not Path(config_dir).exists():
        config_dir = input(
            f"Config directory '{config_dir}' does not exist. "
            "Please enter a valid config directory path: ",
        )
    os.environ["DECKARD_CONFIG_DIR"] = Path(config_dir).resolve().as_posix()

    parsed, _ = parser.parse_known_args()
    module = parsed.module

    sys.argv.pop(sys.argv.index(module))
    if module in SUPPORTED_LAYERS:
        generate_hydra_main(module)
    else:
        raise ValueError(f"Module: {module} not supported. Must be one of {SUPPORTED_LAYERS}")


def generate_hydra_main(layer):
    """Run the parser and main entrypoint for the specified layer via Hydra."""
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
    if hasattr(parsed_args, "overrides") and isinstance(parsed_args.overrides, list):
        # get_args_parser may parse Hydra key=value arguments into `overrides`.
        forwarded_overrides = parsed_args.overrides
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
