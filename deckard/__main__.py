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
    ----
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
    ----
    dict or tuple[dict, ConfigBase]
        If `return_runner` is False, returns the scores as a dictionary.
        If `return_runner` is True, returns a tuple containing the scores and the runner object.

    Raises
    ----
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
    try:
        cfg = OmegaConf.to_container(cfg, resolve=True)
    except ValueError as e:
        if "not an OmegaConf config object" in str(e):
            cfg = cfg.to_dict()
        else:
            raise e
    # Pop "optimizers" from cfg if in cfg and use it to return a subset of the scores
    if "optimizers" in cfg and cfg["optimizers"] and len(cfg["optimizers"]) > 0:
        optimizers = cfg["optimizers"]
        cfg.pop("optimizers")
        if return_runner is not False:
            logger.warning(
                "optimizers can only be used when return_runner is False. Setting return_runner to False.",
            )
    else:
        optimizers = []
    if "directions" in cfg and cfg["directions"] and len(cfg["directions"]) > 0:
        directions = cfg.pop("directions")
        assert len(directions) == len(
            optimizers,
        ), "Length of directions must match length of optimizers."
        for direction in directions:
            assert direction in [
                "minimize",
                "maximize",
                "diff",
            ], "Directions must be either 'minimize' or 'maximize'."
    else:
        directions = []
    if "files" in cfg and len(cfg["files"]) > 0:
        files = cfg.pop("files", {})
    else:
        files = {}
    # Initialize FileConfig
    if isinstance(files, dict):
        files = FileConfig(**files)()
    elif isinstance(files, FileConfig):
        files = files()
    else:
        raise ValueError("files must be a dict or FileConfig instance.")
    files = {**files, **kwargs}
    # Initialize experiment config
    runner = initialize_config(cfg, target=target)
    if isinstance(runner, ExperimentConfig):
        runner.files = FileConfig(**files, experiment_name=runner.experiment_name)
        runner.__post_init__()
        scores = runner.run()
    else:
        scores = runner(*args, **files)
    if optimizers:
        scores = {k: v for k, v in scores.items() if k in optimizers}
        values = list(scores.values())
        if len(directions) > 0:
            attributes = []
            optimize_scores = []
            i = 0
            for direction in directions:
                match direction:
                    case "minimize":
                        optimize_scores.append(values[i])
                    case "maximize":
                        optimize_scores.append(values[i])
                    case "diff":
                        attributes.apppend(values[i])
                i += 1
            if len(optimize_scores) > 0:
                scores = optimize_scores
            else:
                raise ValueError(
                    "No optimization scores found for the specified directions.",
                )
        else:
            scores = values
            attributes = []
        if len(attributes) > 0:
            # TODO: Save these somewhere that optuna can access, but are not used in optimization
            raise NotImplementedError(
                "Experiment attribute tracking not yet implemented.",
            )
    if return_runner is False:
        return scores
    elif return_runner is True:
        return scores, runner
    else:
        raise ValueError("return_runner must be a boolean value.")


def initialize_config(
    cfg: ConfigBase,
    target: str = "deckard.experiment.ExperimentConfig",
    **kwargs,
) -> None:
    """
    ----
    Summary:
        Initializes a configuration object and instantiates a runner based on the provided configuration.

    ----
    Parameters:
        cfg (ConfigBase): The configuration object to be initialized. It is expected to be a dictionary-like object.
        target (str, optional): The default target class path to be used if "_target_" is not already in the configuration.
                                Defaults to "deckard.experiment.ExperimentConfig".
        **kwargs: Additional keyword arguments that may be passed to the instantiation process.

    ----
    Returns:
        None: The function returns None, but it initializes and returns a runner object.
    """
    if "_target_" not in cfg:
        cfg["_target_"] = target
    runner = instantiate(cfg)
    return runner


def parse_optional_args():
    """
    ----
    Parses optional command-line arguments and identifies supported modules.

    This function processes the command-line arguments passed to the script,
    separates optional arguments from module names, and logs the detected
    arguments and modules. It also removes detected modules from `sys.argv`
    to avoid conflicts with other argument parsers.

    ----
    Returns:
        tuple: A tuple containing:
            - optional_args (list): A list of optional arguments that do not
              correspond to supported modules.
            - modules (list): A list of detected module names from the
              command-line arguments.

    ----
    Raises:
        NotImplementedError: If multiple modules are specified in the
        command-line arguments, as only one module is supported at a time.

    ----
    Logs:
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
    ----
    Summary
    ----
    Parses file arguments from a list of optional arguments and filters them
    based on the specified module.

    ----
    Parameters
    ----
    optional_args : list of str
        A list of optional arguments, where some arguments may be in the
        format "key=value".
    module : str
        The name of the module used to filter the parsed file arguments.

    ----
    Returns
    ----
    dict
        A dictionary containing the filtered file arguments, where the keys
        are file names and the values are their corresponding values.

    ----
    Raises
    ----
    None

    ----
    Notes
    ----
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
    ----
    Main entry point for the application.

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
        ValueError("DECKARD_CONFIG_DIR environment variable not set."),
    )
    config_dir = str(Path(config_dir).resolve())
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
    ----
    Summary
    ----
    Handles the default module execution for Deckard by resolving the configuration file
    and initializing the Hydra main function.

    ----
    Parameters
    ----
    config_dir : str
        The directory path where the configuration files are located.

    ----
    Returns
    ----
    None
        Executes the Hydra main function and exits the program if the configuration file
        does not exist.

    ----
    Raises
    ----
    SystemExit
        If the resolved configuration file does not exist.

    ----
    Environment Variables
    ----
    DECKARD_DEFAULT_CONFIG_FILE : str, optional
        Specifies the default configuration file name (default is "default.yaml").
    """
    logger.debug("No optional arguments provided.")
    config_file = os.environ.get("DECKARD_DEFAULT_CONFIG_FILE", "default.yaml")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("Starting Deckard with Hydra configuration.")
    logger.info(f"Config directory: {Path(config_dir).resolve()}")
    config_file = Path(config_dir) / config_file
    logger.info(f"Resolved config file path: {config_file.resolve()}")
    if not config_file.exists():
        logger.error(
            f"Config file {config_file} does not exist. Did you set DECKARD_CONFIG_DIR correctly?",
        )
        sys.exit(1)

    @hydra.main(
        config_path=config_dir,
        config_name=str(config_file.name),
        version_base="1.3",
    )
    def main_hydra(cfg: ExperimentConfig) -> None:
        scores = optimize(cfg=cfg, target="deckard.experiment.ExperimentConfig")
        return scores

    return main_hydra()


def handle_other_modules(config_dir, optional_args, module, args):
    """
    ----
    Summary:
        Handles the execution of a specific module by validating the module, parsing
        optional arguments, and running the optimization process using Hydra.

    ----
    Parameters:
        config_dir (str): The directory path where the configuration files are located.
        optional_args (list): A list of optional arguments passed to the module.
        module (str): The name of the module to be executed.
        args (list): A list of arguments to be passed to the optimization process.

    ----
    Returns:
        Any: The scores returned by the optimization process.

    ----
    Raises:
        SystemExit: If the specified module is not supported.

    ----
    Notes:
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
    if module == "data":
        if "data_config_file" not in files:
            assert "data=" in str(
                optional_args,
            ), "data_config_file argument is required for data module"
            data_arg = [arg for arg in optional_args if arg.startswith("data=")]
            assert (
                len(data_arg) == 1
            ), "data_config_file argument is required for data module"
            module_config_file = data_arg[0].split("=")[1]
        else:
            module_config_file = files["data_config_file"]
    elif module == "model":
        if "model_config_file" not in files:
            assert "model=" in str(
                optional_args,
            ), "model_config_file argument is required for model module"
            model_arg = [arg for arg in optional_args if arg.startswith("model=")]
            assert (
                len(model_arg) == 1
            ), "model_config_file argument is required for model module"
            module_config_file = model_arg[0].split("=")[1]
        else:
            module_config_file = files["model_config_file"]
    elif module == "attack":
        if "attack_config_file" not in files:
            assert "attack=" in str(
                optional_args,
            ), "attack_config_file argument is required for attack module"
            attack_arg = [arg for arg in optional_args if arg.startswith("attack=")]
            assert (
                len(attack_arg) == 1
            ), "attack_config_file argument is required for attack module"
            module_config_file = attack_arg[0].split("=")[1]
        else:
            module_config_file = files["attack_config_file"]
    else:
        raise ValueError(f"Unsupported module: {module}")
    return module_config_file


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
