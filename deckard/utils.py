import logging
from pathlib import Path

import yaml
from hydra import initialize, compose
from hydra.utils import instantiate


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize_config(config_file, params, target) -> object:
    """
    Initializes and composes a Hydra configuration.

    Depending on the provided arguments, this function loads a configuration file,
    applies parameter overrides, and ensures the object_name is set in the configuration.

    Args:
        config_file (str or list or None): Path to the configuration file, or a list of override strings.
        params (list or None): List of parameter overrides in the format ["key=value", ...].
        object_name (str): The object_name string to be included in the configuration if not already present.

    Returns:
        DictConfig: The composed Hydra configuration object.

    Raises:
        AssertionError: If the overrides or params are not provided as lists.
    """
    if config_file and not params:
        logger.info(f"Loading config from {config_file}")
        folder = str(Path(config_file).parent)
        filename = str(Path(config_file).name)
        with initialize(config_path=folder):
            config = compose(config_name=filename)
    elif config_file:
        logger.info(f"Overriding config from {config_file} with params:")
        override_config = config_file
        keys = [k.split("=")[0] for k in override_config]
        for param in params:
            logger.info(f" - {param}")
            override_config.append(param)
        if "_target_" not in keys:
            override_config = [f"++_target_={target}"] + override_config
        assert isinstance(
            override_config, list
        ), "config must be a YAML list of dictionaries"
        if config_file is None:
            with initialize(config_path=None):
                config = compose(config_name=None, overrides=override_config)
        else:
            folder = str(Path(config_file).parent)
            filename = str(Path(config_file).name)
            with initialize(config_path=folder):
                config = compose(config_name=filename, overrides=override_config)
    else:
        params = params if params is not None else []
        keys = [k.split("=")[0] for k in params]
        if "_target_" not in keys:
            params = [f"++_target_={target}"] + params
        with initialize(config_path=None):
            config = compose(config_name=None, overrides=params)
    obj = instantiate(config)
    return obj
