import warnings
import logging
import os
import sys
import json
from pathlib import Path
import yaml
import optuna

from omegaconf import OmegaConf, DictConfig, ListConfig
import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig


from . import LOGGING
from .experiment import ExperimentConfig
from .utils import ConfigBase
from .file import FileConfig

# Setting up some default dictionaries, lists for the CLI
from .file import data_files, model_files, attack_files, all_files
from .layers.compile_results import compile_results_main, compile_results_parser
from .layers.survival import survival_main, survival_parser
module_file_dict = {
    "data": data_files + ["data_config_file"],
    "model": model_files + ["model_config_file"],
    "attack": attack_files + ["attack_config_file"],
    "experiment": all_files,
    "optimize": all_files,
    None: all_files,
}
SUPPORTED_MODULES = list(module_file_dict.keys())

layer_dict = {
    "compile_results" : [compile_results_parser, compile_results_main], 
    "survival" : [survival_parser, survival_main],
}
SUPPORTED_LAYERS = list(layer_dict.keys())

# Set up logging

logger = logging.getLogger(__name__)
logging.config.dictConfig(LOGGING)

# Suppress sklearn runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

assert len(set(SUPPORTED_MODULES + SUPPORTED_LAYERS)) == (len(set(SUPPORTED_MODULES)) + len(set(SUPPORTED_LAYERS)))
# Parse the config_dir argument first to set up config dir


def optimize_multirun(cfg: ConfigBase, hydra_cfg, conf_obj: ExperimentConfig) -> dict:
    """
    Handles optimization in multirun mode.
    
    Parameters
    ----------
    cfg : ConfigBase
        The validated configuration object.
    hydra_cfg : HydraConfig
        The Hydra configuration object.
    conf_obj : ExperimentConfig
        The experiment conf_obj instance.
    
    Returns
    -------
    dict
        The filtered optimization scores.
    """
    assert hasattr(conf_obj, "files"), "conf_obj must have files attribute in multirun mode."
    assert hasattr(conf_obj, "optimizers"), "conf_obj must have optimizers attribute in multirun mode."
    assert hasattr(conf_obj, "directions"), "conf_obj must have directions attribute in multirun mode."
    
    files = prepare_multirun_file_paths(hydra_cfg, conf_obj)
    
    logger.info(f"Saving multirun parameters to {conf_obj.files.params_file}")
    with open(files["params_file"], "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=False), f)
    
    if hasattr(hydra_cfg, "sweeper") and hasattr(hydra_cfg.sweeper, "max_failure_rate"):
        max_failure_rate = hydra_cfg.sweeper.get("max_failure_rate")
    else:
        max_failure_rate = None
    scores = execute_experiment([], conf_obj, files)
    optimizers = conf_obj.optimizers if hasattr(conf_obj, "optimizers") else []
    directions = conf_obj.directions if hasattr(conf_obj, "directions") else []
    filtered_scores, attributes = filter_scores(scores, optimizers, directions)
    
    assert "storage" in hydra_cfg.sweeper, "Storage must be specified in the sweeper config."
    assert "study_name" in hydra_cfg.sweeper, "Study name must be specified in the sweeper config."
    
    storage = hydra_cfg.sweeper.storage
    study_name = hydra_cfg.sweeper.study_name
    study = create_study(study_name, storage, directions, optimizers)
    set_study_metric_names(study=study, optimizers=optimizers)
    set_user_attrs(study=study, attrs=attributes)
    
    logger.info(f"Saving multirun scores to {conf_obj.files.score_file}") 
    with open(files["score_file"], "w") as f:
        json.dump(scores, f, indent=4)
    
    return filtered_scores


def optimize_run(conf_obj: ConfigBase, cfg: ConfigBase, kwargs: dict, args: list = []) -> dict:
    """
    Handles optimization in standard run mode.
    
    Parameters
    ----------
    conf_obj : ConfigBase
        The conf_obj instance.
    cfg : ConfigBase
        The validated configuration object.
    kwargs : dict
        Additional keyword arguments for file configuration.
    args : list, optional
        Additional positional arguments. Default is an empty list.
    
    Returns
    -------
    dict
        The scores from the experiment run.
    """
    if hasattr(cfg, "files"):
        file_dict = OmegaConf.to_container(cfg.files, resolve=True)
    else:
        file_dict = {}
    if hasattr(conf_obj, "experiment_name"):
        file_dict["experiment_name"] = file_dict.get("experiment_name", conf_obj.experiment_name)
    file_dict = {**file_dict, **kwargs}
    if "_target_" not in file_dict:
        files = FileConfig(**file_dict)()
    else:
        files = instantiate(file_dict)()
    if isinstance(conf_obj, ExperimentConfig):
        conf_obj.files = files
        return conf_obj()
    else:
        return conf_obj(*args, **files)


def optimize(
    cfg: ConfigBase,
    target: str,
    args: list = [],
    **kwargs,
) -> dict | tuple[dict, ConfigBase]:
    """
    Parameters
    ----------
    cfg : ConfigBase
        The configuration object to be used for optimization. It is converted
        to a dictionary-like structure for processing.
    target : str
        The target object or function to be optimized.
    args : list, optional
        Additional positional arguments to be passed to the conf_obj. Default is an empty list.

    Returns
    ----------
    dict: returns the scores as a dictionary.

    Notes
    ----
    - If the `cfg` contains an "optimizers" key, the scores are filtered to include
      only those corresponding to the specified optimizers.
    - If the `cfg` contains a "files" key, it is used to initialize a `FileConfig` object.
    - The function initializes an experiment configuration or conf_obj based on the `cfg`
      and executes the optimization process.
    """
    hydra_cfg = HydraConfig.get()
    mode = hydra_cfg.mode
    cfg = validate_cfg(cfg, target)
    conf_obj = instantiate(cfg)
    assert isinstance(conf_obj, ConfigBase), "conf_obj must be an instance of ConfigBase."
    
    if str(mode) == "RunMode.MULTIRUN":
        assert isinstance(conf_obj, ExperimentConfig)
        scores = optimize_multirun(cfg, hydra_cfg, conf_obj)
    else:
        scores = optimize_run(conf_obj, cfg, kwargs, args)
    
    return scores

def validate_cfg(cfg, target):
    dict_ = {}
    dict_["_target_"] = target
    cfg = OmegaConf.merge(OmegaConf.create(dict_), cfg)
    assert hasattr(cfg, "_target_"), "cfg must have a _target_ attribute."
    return cfg

def execute_experiment(conf_obj, *args, **kwargs):
    try:
        scores = conf_obj(*args, **kwargs)
    except Exception as e:
        logger.info(
                f"Experiment failed with error: {e}.",
            )
        if hasattr(conf_obj, "score_dict"):
            scores = conf_obj.score_dict
        else:
            scores = {}
    return scores

def prepare_multirun_file_paths(hydra_cfg, conf_obj):
    conf_obj.experiment_name = f"{hydra_cfg.job.num}"
    conf_obj.__post_init__()
    replace_dict = dict(conf_obj.files.replace)
    replace_dict["num"] = f"{hydra_cfg.job.num}"
    replace_dict["*"] = hash(conf_obj)
    replace_dict["#"] = f"{hydra_cfg.job.num}"
    if hasattr(conf_obj, "data"):
        replace_dict["data_hash"] = hash(conf_obj.data)
    if hasattr(conf_obj, "model"):
        replace_dict["model_hash"] = hash(conf_obj.model)
    if hasattr(conf_obj, "attack"):
        replace_dict["attack_hash"] = hash(conf_obj.attack)
    conf_obj.files.replace = replace_dict
        
        # Set up log, score, and params file paths
    log_dir = Path(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    log_file = log_dir / f"{hydra_cfg.job.name}.log"
    score_file = log_dir / "scores.json"
    params_file = log_dir / "params.yaml"
    conf_obj.files.experiment_name = f"{hydra_cfg.job.num}"
    conf_obj.files.log_file = log_file.as_posix()
    conf_obj.files.score_file = score_file.as_posix()
    conf_obj.files.params_file = params_file.as_posix()
    conf_obj.__post_init__()
    files = conf_obj.files()
    
    return files

def create_study(study_name, storage, directions, optimizers):
    assert len(directions) == len(optimizers), "Length of directions must match length of optimizers."
    if len(directions) == 0:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=directions,
            load_if_exists=True,
            
        )
    return study

def set_study_metric_names(study, optimizers):
    if isinstance(optimizers, ListConfig):
        optimizers = list(optimizers)
    elif isinstance(optimizers, str):
        optimizers = [optimizers]
    elif isinstance(optimizers, tuple):
        optimizers = list(optimizers)
    else:
        raise ValueError(f"optimizers must be a ListConfig, str, or tuple. Got {type(optimizers)}")

    if hasattr(study, "set_metric_names") and len(optimizers) > 0:
        study.set_metric_names(optimizers)

def set_user_attrs(study, attrs):
    if isinstance(attrs, DictConfig):
        attrs = dict(attrs)
    for k, v in attrs.items():
        study.set_user_attr(key=k, value=v)
    

def save_params_file(cfg, files):
    _ = cfg.pop("params", None)
    if "params_file" in files:
        cfg = OmegaConf.create(cfg)
        Path(files["params_file"]).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, files["params_file"])
    else:
        raise ValueError("params_file must be specified in files to save parameters.")
    return cfg



def filter_scores(scores: dict, optimizers: list, directions: list) -> dict:
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
        return scores, {}
    other_scores = {k: v for k, v in scores.items() if k not in optimizers}
    scores = {k: v for k, v in scores.items() if k in optimizers}
    missing_scores = set(optimizers) - set(scores.keys())
    values = list(scores.values())
    if directions:
        assert len(directions) == len(
            optimizers,
        ), f"Length of directions must match length of optimizers. Got {len(directions)} and {len(optimizers)}."
        optimize_scores = []
        attributes = {}
        for i, direction in enumerate(directions):
            key = optimizers[i]
            if key in missing_scores:
                if direction == "diff":
                    attributes[key] = None
                elif direction == "minimize":
                    optimize_scores.append(float("inf"))
                elif direction == "maximize":
                    optimize_scores.append(float("-inf"))

                else:
                    raise ValueError(f"Invalid direction: {direction}")
            else:
                if direction in ["minimize", "maximize"]:
                    optimize_scores.append(scores[key])
                elif direction == "diff":
                    attributes[key] = scores[key]
                else:
                    raise ValueError(f"Invalid direction: {direction}")
        if not optimize_scores:
            raise ValueError("No optimization scores found for the specified directions.")
        if len(missing_scores) > 0:
            logger.error(f"Missing scores: {missing_scores}")
        values = optimize_scores
    else:
        attributes = {}
    attributes.update(other_scores)
    values = tuple(values)
    if isinstance(values, (tuple, list)) and len(values) == 1:
        values = values[0]
    logger.info(f"Optimization values: {values}")
    logger.info(f"Experiment attributes: {attributes}")
    return values, attributes


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
    for m in SUPPORTED_MODULES:
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
        if module in SUPPORTED_LAYERS:
            handle_layers(module)
        elif module in [None, "experiment", "optimize"]:
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
    working_dir = os.getcwd()
    logger.info(f"Current working directory: {working_dir}")
    logger.info("Starting Deckard with Hydra configuration.")
    logger.info(f"Config directory: {Path(config_dir).resolve()}")
    config_file = Path(config_dir) / config_file
    # Make sure config_dir is relatve to working dir
    if not Path(config_dir).is_absolute():
        config_dir = os.path.relpath(config_dir, working_dir)
    config_file = Path(config_dir) / config_file.name
    logger.info(f"Resolved config file path: {config_file}")
    if not Path(config_file).exists():
        logger.error(
            f"Config file {config_file} does not exist. Did you set DECKARD_CONFIG_DIR correctly?",
        )
        raise FileNotFoundError(config_file)

    @hydra.main(
        config_path=str(Path(config_dir).resolve()),
        config_name=str(config_file.name),
        version_base="1.3",
    )
    def main_hydra(cfg: ExperimentConfig) -> None:
        scores = optimize(cfg=cfg, target="deckard.experiment.ExperimentConfig")
        return scores

    return main_hydra()




def handle_layers(layer):
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
