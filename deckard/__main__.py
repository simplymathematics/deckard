import warnings
import logging
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate


from .file import FileConfig, data_files, model_files, attack_files, all_files
from .experiment import ExperimentConfig
from .utils import ConfigBase

module_file_dict = {
    "data": data_files + ["data_config_file"],
    "model": model_files + ["model_config_file"],
    "attack": attack_files + ["attack_config_file"],
    "experiment": all_files,
    "optimize": all_files,
    None: all_files,
}


logger = logging.getLogger(__name__)


# Suppress sklearn runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

supported_modules = ["data", "model", "attack", "experiment", "optimize", None]
# Parse the config_dir argument first to set up config dir


def optimize(
    cfg: ConfigBase,
    target: str,
    return_runner: bool = False,
    args: list = [],
    **kwargs: dict,
) -> dict | tuple[dict, ConfigBase]:
    """
    Parameters
    ----------
    cfg : ConfigBase
        The configuration object to be used for optimization. It is converted
        to a dictionary-like structure for processing.
    target : str
        The target object or function to be optimized.
    return_runner : bool, optional
        If True, the function returns both the scores and the runner object.
        If False, only the scores are returned. Default is False.
    args : list, optional
        Additional positional arguments to be passed to the runner. Default is an empty list.
    **kwargs : dict
        Additional keyword arguments to be merged with the file configuration.

    Returns
    ----------
    dict or tuple[dict, ConfigBase]
        If `return_runner` is False, returns the scores as a dictionary.
        If `return_runner` is True, returns a tuple containing the scores and the runner object.

    Raises
    ----------
    ValueError
        If `return_runner` is not a boolean value.

    Notes
    ----
    - If the `cfg` contains an "optimizers" key, the scores are filtered to include
      only those corresponding to the specified optimizers.
    - If the `cfg` contains a "files" key, it is used to initialize a `FileConfig` object.
    - The function initializes an experiment configuration or runner based on the `cfg`
      and executes the optimization process.
    """
    cfg = _convert_config_to_dict(cfg)
    optimizers, directions = _extract_optimizers_and_directions(cfg, return_runner)
    files = _initialize_files(cfg, kwargs)
    runner = initialize_config(cfg, target=target)
    scores = _run_experiment(runner, files, args)
    scores = _filter_scores(scores, optimizers, directions)
    if return_runner:
        return scores, runner
    else:
        return scores


def _convert_config_to_dict(cfg: ConfigBase) -> dict:
    """
    Converts a configuration object to a dictionary.
    
    Parameters
    ----------
        -  cfg (ConfigBase): The configuration object to be converted.
    Returns
    ----------
        dict: A dictionary representation of the configuration object.
    Raises
    ----------
        ValueError: If the input is not an OmegaConf config object and cannot be converted to a dictionary.
    """
    try:
        return OmegaConf.to_container(cfg, resolve=True)
    except ValueError as e:
        if "not an OmegaConf config object" in str(e):
            return cfg.to_dict()
        raise e


def _extract_optimizers_and_directions(cfg: dict, return_runner: bool) -> tuple[list, list]:
    """
    Overview
    --------
    Extracts the "optimizers" and "directions" from the provided configuration dictionary
    and validates their consistency.

    Parameters
    ---
    cfg : dict
        The configuration dictionary from which "optimizers" and "directions" are extracted.
    return_runner : bool
        A boolean flag indicating whether the runner object should be returned.

    Returns
    -------
    tuple[list, list]
        A tuple containing two lists:
        - The first list contains the extracted optimizers.
        - The second list contains the extracted directions.

    Raises
    -------
        AssertionError
            - If the length of "directions" does not match the length of "optimizers".
            - If an invalid direction is provided (not one of "minimize", "maximize", or "diff").

    Notes
    -------
    - If "optimizers" are present in the configuration and `return_runner` is True, a warning
      is logged, and `return_runner` is effectively set to False.
    - The "directions" list must match the length of the "optimizers" list and can only
      contain valid values ("minimize", "maximize", or "diff").
    """
    optimizers = cfg.pop("optimizers", []) if "optimizers" in cfg else []
    if optimizers and return_runner:
        logger.warning(
            "optimizers can only be used when return_runner is False. Setting return_runner to False.",
        )
    directions = cfg.pop("directions", []) if "directions" in cfg else []
    if directions:
        assert len(directions) == len(
            optimizers,
        ), "Length of directions must match length of optimizers."
        for direction in directions:
            assert direction in ["minimize", "maximize", "diff"], "Invalid direction."
    return optimizers, directions


def _initialize_files(cfg: dict, kwargs: dict) -> dict:
    """
    Overview
    ---------
    Initializes file configurations from the provided configuration dictionary.
    
    Parameters
    ----------
        - cfg (dict): The configuration dictionary, which may contain a "files" key.
        - kwargs (dict): Additional keyword arguments to merge with the file configurations.
    Returns
    ----------
        dict: A dictionary containing the merged file configurations and additional arguments.
    Raises
    ----------
        ValueError: If the "files" key in the configuration is not a dictionary or a FileConfig instance.
    """
    files = cfg.pop("files", {}) if "files" in cfg else {}
    if isinstance(files, dict):
        files = FileConfig(**files)()
    elif isinstance(files, FileConfig):
        files = files()
    else:
        raise ValueError("files must be a dict or FileConfig instance.")
    return {**files, **kwargs}


def _run_experiment(runner: ConfigBase, files: dict, args: list) -> dict:
    """
    Overview
    --------
        Executes an experiment or a runner function based on the provided configuration.

    Parameters
    -----------
        runner : ConfigBase
            The runner object or configuration to execute. If it is an instance of
            `ExperimentConfig`, it will be initialized and executed as an experiment.
        files : dict
            A dictionary containing file configurations to be passed to the runner.
        args : list
            A list of additional arguments to be passed to the runner.

    Returns
    -------
        dict
            The results or scores of the experiment or runner execution.

    Raises
    -------
        None

    Notes
    -------
        - If the `runner` is an instance of `ExperimentConfig`, it initializes the
        experiment files and calls its `run` method.
        - If the `runner` is not an `ExperimentConfig`, it is treated as a callable
        and executed with the provided `args` and `files`.
    """
    if isinstance(runner, ExperimentConfig):
        runner.files = FileConfig(**files, experiment_name=runner.experiment_name)
        runner.__post_init__()
        return runner()
    return runner(*args, **files)


def _filter_scores(scores: dict, optimizers: list, directions: list) -> dict:
    """
    Overview
    ---
    Filters and processes the scores dictionary based on the specified optimizers
    and directions.

    Parameters
    ----------
    scores : dict
        A dictionary containing the scores to be filtered and processed.
    optimizers : list
        A list of optimizer names to filter the scores. If empty, all scores are returned.
    directions : list
        A list of directions ("minimize", "maximize", or "diff") corresponding to the
        optimizers. Used to further process the filtered scores.

    Returns
    -------
    dict
        A dictionary containing the filtered and processed scores.

    Raises
    -------
    ValueError
        - If the length of `directions` does not match the length of `optimizers`.
        - If an invalid direction is provided.
        - If no optimization scores are found for the specified directions.

    Notes
    -------
    - If `optimizers` is empty, the function returns the original `scores` dictionary.
    - The `directions` parameter is used to determine how the scores are processed:
        - "minimize" or "maximize": Adds the score to the optimization scores.
        - "diff": Adds the score to the attributes.
    - If no valid optimization scores are found, a `ValueError` is raised.
    """
    if not optimizers:
        return scores
    scores = {k: v for k, v in scores.items() if k in optimizers}
    values = list(scores.values())
    if directions:
        assert len(directions) == len(
            optimizers,
        ), f"Length of directions must match length of optimizers. Got {len(directions)} and {len(optimizers)}."
        optimize_scores = []
        attributes = []
        for i, direction in enumerate(directions):
            match direction:
                case "minimize" | "maximize":
                    optimize_scores.append(float(values[i]))
                case "diff":
                    attributes.append(float(values[i]))
        if optimize_scores:
            return optimize_scores
        raise ValueError("No optimization scores found for the specified directions.")
    if len(attributes) > 0:
        raise NotImplementedError("Storing metrics not used for optimization not yet implemented.")
    values = tuple(values)
    return values




def initialize_config(
    cfg: ConfigBase,
    target: str = "deckard.experiment.ExperimentConfig",
    **kwargs,
) -> None:
    """
    Overview
    ----------
        Initializes a configuration object and instantiates a runner based on the provided configuration.

    Parameters
    ----------
    cfg (ConfigBase): The configuration object to be initialized. It is expected to be a dictionary-like object.
        target (str, optional): The default target class path to be used if "_target_" is not already in the configuration.
                                Defaults to "deckard.experiment.ExperimentConfig".
        **kwargs: Additional keyword arguments that may be passed to the instantiation process.

    Returns:
    --------
        None: The function returns None, but it initializes and returns a runner object.
    """
    if "_target_" not in cfg:
        cfg["_target_"] = target
    for k, v in kwargs.items():
        cfg[k] = v
    runner = instantiate(cfg)
    return runner


def parse_optional_args():
    """
    Overview
    --------
    Parses optional command-line arguments and identifies supported modules.

    This function processes the command-line arguments passed to the script,
    separates optional arguments from module names, and logs the detected
    arguments and modules. It also removes detected modules from `sys.argv`
    to avoid conflicts with other argument parsers.

    Returns
    ----------
        tuple: A tuple containing:
            - optional_args (list): A list of optional arguments that do not
              correspond to supported modules.
            - modules (list): A list of detected module names from the
              command-line arguments.

    Raises
    ----------
        NotImplementedError: If multiple modules are specified in the
        command-line arguments, as only one module is supported at a time.

    Logs
    ----------
        - Command-line arguments if any are provided.
        - All optional arguments after processing.
        - Detected modules in the optional arguments.
    """
    args = sys.argv[1:]
    if len(args) > 0:
        logger.info(f"Command-line arguments: {args}")
    optional_args = [arg for arg in args if not arg.startswith("--")]
    modules = []
    for m in supported_modules:
        if m in optional_args:
            for opt_arg in optional_args:
                if opt_arg == m:
                    # Find the index of the module in optional_args
                    i = optional_args.index(m)
                    module = optional_args.pop(i)
                    modules.append(module)
    if len(optional_args) > 0:
        logger.info(f"All optional arguments: {optional_args}")
    if len(modules) > 0:
        logger.info(f"Detected modules in optional arguments: {modules}")
    elif len(modules) >= 1:
        raise NotImplementedError(
            "Multiple modules specified in command-line arguments. Only one module at a time is supported.",
        )
    # Remove modules from sys.argv to prevent Hydra from complaining about unknown arguments
    for module in modules:
        sys.argv.remove(module)
    return optional_args, modules


def parse_files_from_optional_args(optional_args, module):
    """
    Overview
    ---------
    Parses file arguments from a list of optional arguments and filters them
    based on the specified module.

    Parameters
    ----------
    optional_args : list of str
        A list of optional arguments, where some arguments may be in the
        format "key=value".
    module : str
        The name of the module used to filter the parsed file arguments.

    Returns
    -------
    dict
        A dictionary containing the filtered file arguments, where the keys
        are file names and the values are their corresponding values.

    Raises
    -------
    None

    Notes
    ------
    - Arguments starting with "~", "++", or "+" will have these prefixes
      stripped from their keys.
    - Only files that are part of the module's file list (as defined in
      `module_file_dict`) will be included in the returned dictionary.
    - The function logs the parsed file arguments and validates the final
      subset of files.
    """
    files = {}
    pop_these = []
    for arg in optional_args:
        try:
            k, v = arg.split("=")
            pop_these.append(arg)
            k = k.lstrip("~")
            k = k.lstrip("++")
            k = k.lstrip("+")
            files[k] = v
        except ValueError:  # raised when there is no "=" in arg
            pass
    logger.info(f"Parsed file arguments from optional args: {files}")
    module_files = module_file_dict.get(module, [])
    file_subset = {}
    for k, v in files.items():
        if k in module_files:
            file_subset[k] = v
    files = file_subset
    validate_files(files.keys(), module_files, module)
    return files


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
        None,
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
    optional_args, modules = parse_optional_args()
    if len(modules) >= 1:
        pass
    else:
        modules = [None]
    args = []
    for module in modules:
        if module in [None, "experiment", "optimize"]:
            handle_default_module(config_dir)
        else:
            handle_other_modules(config_dir, optional_args, module, args)


def handle_default_module(config_dir):
    """
    Overview
    ----
    Handles the default module execution for Deckard by resolving the configuration file
    and initializing the Hydra main function.

    Parameters
    -----------
    config_dir : str
        The directory path where the configuration files are located.

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
    DECKARD_DEFAULT_CONFIG_FILE : str, optional
        Specifies the default configuration file name (default is "default.yaml").
    """
    logger.debug("No optional arguments provided.")
    config_file = os.environ.get("DECKARD_DEFAULT_CONFIG_FILE", "default.yaml")
    working_dir = os.getcwd
    logger.info(f"Current working directory: {working_dir()}")
    logger.info("Starting Deckard with Hydra configuration.")
    logger.info(f"Config directory: {Path(config_dir).resolve()}")
    config_file = Path(config_dir) / config_file
    # Make config_dir is relatve to working dir
    if not Path(config_dir).is_absolute():
        config_dir = os.path.relpath(config_dir, working_dir())
    config_file = Path(config_dir) / config_file.name
    logger.info(f"Resolved config file path: {config_file}")
    if not Path(config_file).exists():
        logger.error(
            f"Config file {config_file} does not exist. Did you set DECKARD_CONFIG_DIR correctly?",
        )
        sys.exit(1)

    @hydra.main(
        config_path=str(Path(config_dir).resolve()),
        config_name=str(config_file.name),
        version_base="1.3",
    )
    def main_hydra(cfg: ExperimentConfig) -> None:
        scores = optimize(cfg=cfg, target="deckard.experiment.ExperimentConfig")
        return scores

    return main_hydra()


def handle_other_modules(config_dir, optional_args, module, args):
    """
    Overview
    --------
        Handles the execution of a specific module by validating the module, parsing
        optional arguments, and running the optimization process using Hydra.

    Parameters:
    -----------
        config_dir (str): The directory path where the configuration files are located.
        optional_args (list): A list of optional arguments passed to the module.
        module (str): The name of the module to be executed.
        args (list): A list of arguments to be passed to the optimization process.

    Returns
    -------
        Any: The scores returned by the optimization process.

    Raises
    -------
        SystemExit: If the specified module is not supported.

    Notes
    ------
        - The function validates the module and its associated files before execution.
        - It uses Hydra to manage configuration and execute the optimization process.
        - Unsupported modules will result in an error log and termination of the program.
    """
    logger.info(f"Optional args after module: {optional_args}")
    files = parse_files_from_optional_args(optional_args, module)
    if module not in supported_modules:
        logger.error(
            f"Unsupported module: {module}. Supported modules are: {supported_modules}",
        )
        sys.exit(1)
    module_config_file = validate_module_and_files(module, files, optional_args)

    @hydra.main(
        config_path=str(Path(config_dir)),
        config_name=str(Path(module, module_config_file)),
        version_base="1.3",
    )
    def main_hydra(cfg: ConfigBase) -> None:
        sub_dict = cfg.get(module)
        # pop {module}_config_file from files to avoid passing it to optimize
        files.pop(f"{module}_config_file", None)
        scores, runner = optimize(
            cfg=sub_dict,
            target=f"deckard.{module}.{module.capitalize()}Config",
            return_runner=True,
            args=args,
            **files,
        )
        args.append(runner)
        return scores

    return main_hydra()


def validate_module_and_files(module, files, optional_args=None):
    """
    Validate the provided module and its associated configuration files.

    ----param module: The name of the module to validate.
                      Supported values are "data", "model", and "attack".
    ----type module: str
    ----param files: A dictionary containing configuration file paths for the module.
                     The dictionary must include specific keys based on the module:
                     - "data": Requires "data_config_file".
                     - "model": Requires "model_config_file".
                     - "attack": Requires "attack_config_file".
    ----type files: dict
    ----raises ValueError: If the module is unsupported or the required configuration
                           file key is missing in the files dictionary.
    ----return: The path to the module's configuration file.
    ----rtype: str
    """
    required_file_key = {
        "data": "data_config_file",
        "model": "model_config_file",
        "attack": "attack_config_file",
    }

    if module not in required_file_key:
        raise ValueError(f"Unsupported module: {module}")

    config_key = required_file_key[module]

    if config_key in files:
        return files[config_key]

    assert f"{module}=" in str(
        optional_args,
    ), f"{config_key} argument is required for {module} module"

    module_arg = [arg for arg in optional_args if arg.startswith(f"{module}=")]
    assert (
        len(module_arg) == 1
    ), f"{config_key} argument is required for {module} module"

    return module_arg[0].split("=")[1]


def validate_files(files, supported_files, module_name):
    """
    ----
    Validates a list of files against a set of supported files for a specific module.

    Parameters
    ----
    files : list
        A list of file names to validate.
    supported_files : list
        A list of supported file names for the module.
    module_name : str
        The name of the module being validated.

    Raises
    ----
    SystemExit
        If any file in the `files` list is not in the `supported_files` list,
        logs an error message and exits the program.
    """
    for file in files:
        if file not in supported_files:
            logger.error(
                f"Unsupported {module_name} file argument: {file}. Supported {module_name} file arguments are: {supported_files}",
            )
            sys.exit(1)


if __name__ == "__main__":

    main()
