import argparse
import warnings
import logging
import os
import sys
from pathlib import Path

from hydra import initialize_config_dir


from .experiment import ExperimentConfig
from .file import FileConfig
from .utils import create_parser_from_function, initialize_config

logger = logging.getLogger(__name__)

# Get env variables for logging level, default to INFO
log_level = os.getenv("DECKARD_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")



def main():
    init_parser = argparse.ArgumentParser(
        description="Deckard Experiment Initialization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    init_parser.add_argument_group("Experiment Initialization Arguments", "Experiment initialization parameters.")
    # init_parser = create_parser_from_function(ExperimentConfig.__init__, init_parser)
    init_parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment.",
    )
    init_parser.add_argument(
        "--config_dir",
        type=str,
        default="config",
        help="Directory for configuration files.",
        
    )
    init_parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        help="Additional file parameters to override in the format key=value.",
        metavar="key=value",
        default=None,
    )
    # Data config
    data_args = init_parser.add_argument_group("Data Configuration", "Parameters for data configuration.")
    data_args.add_argument(
        "--data_config_file",
        type=str,
        required=True,
        help="Path to the data configuration YAML file.",
    )
    data_args.add_argument(
        "--data_config_dir",
        type=str,
        default="data",
        help="Directory for data configuration files.",
    )
    
    # Model config
    model_args = init_parser.add_argument_group("Model Configuration", "Parameters for model configuration.")
    model_args.add_argument(
        "--model_config_file",
        type=str,
        required=False,
        default=None,
        help="Path to the model configuration YAML file.",
    )
    model_args.add_argument(
        "--model_config_dir",
        type=str,
        default="model",
        help="Directory for model configuration files.",
    )
    # Defense config
    defense_args = init_parser.add_argument_group("Defense Configuration", "Parameters for defense configuration.")
    defense_args.add_argument(
        "--defense_config_file",
        type=str,
        required=False,
        default=None,
        help="Path to the defense configuration YAML file.",
    )
    defense_args.add_argument(
        "--defense_config_dir",
        type=str,
        default="defend",
        help="Directory for defense configuration files.",
    )
    
    # Attack config
    attack_args = init_parser.add_argument_group("Attack Configuration", "Parameters for attack configuration.")
    attack_args.add_argument(
        "--attack_config_file",
        type=str,
        required=False,
        default=None,
        help="Path to the attack configuration YAML file.",
    )
    attack_args.add_argument(
        "--attack_config_dir",
        type=str,
        default="attack",
        help="Directory for attack configuration files.",
    )

    args = init_parser.parse_args()
    working_dir = os.getcwd()
    # Initialize config directory
    # Assert that args.config_dir exists
    assert Path(args.config_dir).exists(), f"Config directory {args.config_dir} does not exist."
    config_dir = str(Path(args.config_dir).resolve())
    initialize_config_dir(config_dir)

    logger.info(f"Current working directory: {working_dir}")
    for group in init_parser._action_groups:
        if group.title == "Data Configuration":
            data_args = group
            data_config_file = getattr(args, "data_config_file")
            # Assume ./config_dir/data_config_dir/data_config_file
            config_file = Path(getattr(args, "data_config_dir")) / data_config_file
            logger.info(f"Loading data_config from resolved path: {config_file.resolve()}")
            if config_file.is_absolute():
                # Use os to make this relative to working dir
                config_file = Path(os.path.relpath(config_file, working_dir))
                logger.info(f"Converted absolute config_file to relative path: {config_file}")
            data = initialize_config(config_file=config_file, params={}, target="deckard.data.DataConfig")
        elif group.title == "Model Configuration":
            model_args = group
            model_config_file = getattr(args, "model_config_file")
            if model_config_file is None:
                model = None
                continue
            config_file = Path(getattr(args, "model_config_dir")) / model_config_file
            logger.info(f"Loading model_config from resolved path: {config_file.resolve()}")
            if config_file.is_absolute():
                # Use os to make this relative to working dir
                config_file = Path(os.path.relpath(config_file, working_dir))
                logger.info(f"Converted absolute config_file to relative path: {config_file}")
            model = initialize_config(config_file=config_file, params={}, target="deckard.model.ModelConfig")
        elif group.title == "Defense Configuration":
            defense_args = group
            defense_config_file = getattr(args, "defense_config_file")
            if defense_config_file is None:
                defense = None
                continue
            config_file = Path(getattr(args, "defense_config_dir")) / defense_config_file
            logger.info(f"Loading defense_config from resolved path: {config_file.resolve()}")
            defense = initialize_config(config_file=config_file, params={}, target="deckard.defense.DefenseConfig")
            model = defense
        elif group.title == "Attack Configuration":
            attack_args = group
            attack_config_file = getattr(args, "attack_config_file")
            if attack_config_file is None:
                attack = None
                continue
            config_file = Path(getattr(args, "attack_config_dir")) / attack_config_file
            logger.info(f"Loading attack_config from resolved path: {config_file.resolve()}")
            attack = initialize_config(config_file=config_file, params={}, target="deckard.attack.AttackConfig")
    # File config
    files = FileConfig(
        **vars(args).copy()
    )
    files = files(**{k.split("=")[0]: v for k, v in (arg.split("=") for arg in args.files)}) if args.files else files()
    experiment = ExperimentConfig(
        experiment_name=args.experiment_name,
        data=data,
        model=model,
        attack=attack,
        files=files,
    )
    scores = experiment(**files)
if __name__ == "__main__":
    main()



