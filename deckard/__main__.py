import warnings
import logging
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate


from .data import DataConfig
from .model import ModelConfig
from .attack import AttackConfig
from .file import FileConfig, data_files, model_files, attack_files
from .experiment import ExperimentConfig
from .utils import ConfigBase



logger = logging.getLogger(__name__)


# Suppress sklearn runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

supported_modules = ["data", "model", "attack"]
# Parse the config_dir argument first to set up config dir


def optimize(cfg: ConfigBase, target, **kwargs) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    # Pop "optimizers" from cfg if in cfg and use it to return a subset of the scores
    if "optimizers" in cfg and cfg.optimizers is not None:
        optimizers = cfg.optimizers
        cfg.pop("optimizers")
    else:
        optimizers = None
    if "files" in cfg and len(cfg["files"]) > 0:
        files = cfg.pop("files")
        files = FileConfig(**files)()
    else:
        files = {}
    files = {**files, **kwargs}
    # Initialize experiment config
    runner = initialize_config(cfg, target=target)
    if isinstance(runner, ExperimentConfig):
        runner.files = FileConfig(**files, experiment_name=runner.experiment_name)
        runner.__post_init__()
        scores = runner.run()
    else:
        scores = runner(**files)
    if optimizers:
        scores = {k: v for k, v in scores.items() if k in optimizers}
    return scores

def initialize_config(cfg: ConfigBase, target: str = "deckard.experiment.ExperimentConfig", **kwargs) -> None:
    if "_target_" not in cfg:
        cfg["_target_"] = target
    runner = instantiate(cfg)
    return runner


def main():
    # Get config dir from environment variable if set
    config_dir = os.environ.get(
        "DECKARD_CONFIG_DIR",
        ValueError("DECKARD_CONFIG_DIR environment variable not set."),
    )
    config_dir = str(Path(config_dir).resolve())
    
    args = sys.argv[1:]
    if len(args) > 0:
        logger.info(f"Command-line arguments: {args}")
        
    optional_args = sys.argv[1:]
    i = 0
    for m in supported_modules:
        if m in optional_args:
            module = m
            optional_args.pop(i)
            break
        i += 1
    else:
        module = None
    if len(optional_args) > 0:
        logger.info(f"All optional arguments: {optional_args}")
    if module is None:
        logger.debug("No optional arguments provided.")
        config_file = os.environ.get("DECKARD_DEFAULT_CONFIG_FILE", "default.yaml")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info("Starting Deckard with Hydra configuration.")
        logger.info(f"Config directory: {Path(config_dir).resolve()}")
        config_file = Path(config_dir) / config_file
        logger.info(f"Resolved config file path: {config_file.resolve()}")
        if not config_file.exists():
            logger.error(f"Config file {config_file} does not exist. Did you set DECKARD_CONFIG_DIR correctly?")
            sys.exit(1)
        @hydra.main(
            config_path=config_dir,
            config_name=str(config_file.name),
            version_base="1.3",
        )
        def main_hydra(cfg: ExperimentConfig) -> None:
            scores = optimize(cfg=cfg, target="deckard.experiment.ExperimentConfig")
            return scores
    else:
        logger.info(f"Optional args after module: {optional_args}")
        # Remove the first N entries of sys.argv where N is len(optional_args) + 1 (for module
        sys.argv = sys.argv[len(optional_args) :]
        # Turn other optional args into key-value pairs for hydra overrides
        files = {}
        for arg in optional_args:
            try:
                k,v = arg.split("=")
                k = k.lstrip("~")
                k = k.lstrip("++")
                k = k.lstrip("+")
                files[k] = v
            except ValueError:
                pass
        if module not in supported_modules:
            logger.error(f"Unsupported module: {module}. Supported modules are: {supported_modules}")
            sys.exit(1)
        if module == "data":
            module_config_file = files.pop("data_config_file", ValueError("data_config_file argument is required for data module"))
            for file in files:
                if file not in data_files:
                    logger.error(f"Unsupported data file argument: {file}. Supported data file arguments are: {data_files}")
                    sys.exit(1)
        elif module == "model":
            module_config_file = files.pop("model_config_file", ValueError("model_config_file argument is required for model module"))
            for file in files:
                if file not in model_files:
                    logger.error(f"Unsupported model file argument: {file}. Supported model file arguments are: {model_files}")
                    sys.exit(1)
        elif module == "attack":
            module_config_file = files.pop("attack_config_file", ValueError("attack_config_file argument is required for attack module"))
            for file in files:
                if file not in attack_files:
                    logger.error(f"Unsupported attack file argument: {file}. Supported attack file arguments are: {attack_files}")
                    sys.exit(1)
        else:
            raise ValueError(f"Unsupported module: {module}")
        @hydra.main(
            config_path=str(Path(config_dir)),
            config_name=str(Path(module, module_config_file)),
            version_base="1.3",
        )
        
        def main_hydra(cfg: DataConfig) -> None:
            scores = optimize(cfg=cfg.get(module), **files, target = f"deckard.{module}.{module.capitalize()}Config")
            return scores
    return main_hydra()
            
            
            
            
            
        


if __name__ == "__main__":

    main()
