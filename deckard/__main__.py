
import logging
import os
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path


from . import LOGGING

from .layers import SUPPORTED_LAYERS, layer_dict
from .experiment import ExperimentConfig



# Set up logging
logger = logging.getLogger(__name__)
logging.config.dictConfig(LOGGING)


def get_configuration_paths():
    """
    """
    # Get config dir from environment variable if set
    config_dir = os.environ.get(
        "DECKARD_CONFIG_DIR",
        "config",
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
    config_file = Path(os.environ.get("DECKARD_DEFAULT_CONFIG_FILE", "default.yaml")).as_posix()
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

def main():
    """
    Main entry point for the application.
    Overview
    ---------
    This function retrieves the configuration directory from the environment variable
    `DECKARD_CONFIG_DIR`, resolves its absolute path, and processes optional arguments
    and modules. Depending on the specified modules, it delegates handling to appropriate
    functions.

    ----
    Raises
    ------
    ValueError
        If the `DECKARD_CONFIG_DIR` environment variable is not set.

    ----
    Notes
    -----
    - If no modules are specified, a default module is used.
    - Handles specific modules ("experiment", "optimize") differently from other modules.

    ----
    """
    # Get config dir from environment variable if set
    config_dir = os.environ.get(
        "DECKARD_CONFIG_DIR",
        "config",
    )
    if config_dir is None:
        config_dir = input("Please enter the config directory path: ")
    while not Path(config_dir).exists():
        # Deckard_config dir does not exist, have the user set it using input()
        config_dir = input(
            f"The provided config directory path '{config_dir}' does not exist. Please enter a valid config directory path: ",
        )
        # Prompt user to confirm the path exists
        if not Path(config_dir).exists():
            config_dir = None
    # Set the environment variable for future use
    os.environ["DECKARD_CONFIG_DIR"] = Path(config_dir).resolve().as_posix()
    # strip the username from the path for logging
    module = sys.argv[1]
    sys.argv.pop(1)
    if module in [None, "experiment", "optimize"]:
        handle_default_module()
    elif module in SUPPORTED_LAYERS:
        handle_other_layers(module)
    else:
        raise ValueError(f"Module: {module} not supported. Must be one of {SUPPORTED_LAYERS}")


def handle_default_module():
    """
    Overview
    ----
    Handles the default module execution for Deckard by resolving the configuration file
    and initializing the Hydra main function.

    Returns
    -------
    None
        Executes the Hydra main function and exits the program if the configuration file
        does not exist.

    Raises
    --------
    SystemExit
        If the resolved configuration file does not exist.

    Environment Variables
    ---------------------
    DECKARD_DEFAULT_CONFIG_FILE : str mandatory
        Specifies the default configuration file name (default is "default.yaml").
    DECKARD_CONFIG_DIR : str, mandatory
    """
    config_dir, config_file = get_configuration_paths()
    @hydra.main(
        config_path=str(Path(config_dir).resolve()),
        config_name=config_file,
        version_base="1.3",
    )
    def main_hydra(cfg: ExperimentConfig) -> None:
        optimize_main = layer_dict["optimize"][1]
        scores = optimize_main(cfg=cfg)
        return scores

    return main_hydra()

def handle_other_layers(layer):
    """Run the parser and main entrypoint for the specified layer via Hydra."""
    if layer not in layer_dict:
        logger.error(f"Unsupported layer: {layer}. Supported layers are: {list(layer_dict)}")
        raise ValueError

    parser, main_fn = layer_dict[layer]
    if not hasattr(parser, "parse_known_args"):
        raise ValueError("Parser object does not have .parse_known_args")

    # Parse layer-specific args first, then leave remaining args for Hydra.
    parsed_args, hydra_args = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *hydra_args]
    @hydra.main(
        config_path=None,
        config_name=None,
        version_base="1.3",
    )
    def main_hydra(cfg: DictConfig) -> None:
        args = vars(parsed_args).copy()

        # Allow Hydra overrides for parser keys when present.
        if cfg is not None:
            for key in list(args.keys()):
                if key in cfg and cfg[key] is not None:
                    args[key] = cfg[key]
                return main_fn(**args)

    return main_hydra()

if __name__ == "__main__":

    main()
