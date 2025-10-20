import argparse
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .data import DataConfig
from .model import ModelConfig
from .model.defend import DefenseConfig
from .attack import AttackConfig
from .file import FileConfig
from .score import ScorerDictConfig
from .experiment import ExperimentConfig
from .utils import create_parser_from_function

logger = logging.getLogger(__name__)

supported_modules = ["data", "model", "attack", "defense", "score"]


# For each module, create a parser and main function mapping
# module_init_parsers = {
#     "data": (create_parser_from_function(DataConfig.__init__, skip_params=["self"]), DataConfig),
#     "model": (create_parser_from_function(ModelConfig.__init__, skip_params=["self"]), ModelConfig),
#     "attack": (create_parser_from_function(AttackConfig.__init__, skip_params=["self"]), AttackConfig),
#     "defense": (create_parser_from_function(DefenseConfig.__init__, skip_params=["self"]), DefenseConfig),
#     "score": (create_parser_from_function(ScorerDictConfig.__init__, skip_params=["self"]), ScorerDictConfig),
# }

module_call_parsers = {
    "data": (create_parser_from_function(DataConfig.__call__), DataConfig),
    "model": (
        create_parser_from_function(ModelConfig.__call__, exclude=["data"]),
        ModelConfig,
    ),
    "attack": (
        create_parser_from_function(
            AttackConfig.__call__,
            exclude=["data", "estimator"],
        ),
        AttackConfig,
    ),
    "defense": (
        create_parser_from_function(DefenseConfig.__call__, exclude=["data"]),
        DefenseConfig,
    ),
    "score": (
        create_parser_from_function(ScorerDictConfig.__call__, exclude=["data"]),
        ScorerDictConfig,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Deckard Will Run Your Experiment",
        usage=f"python -m deckard <module>  <module>  --files [<args>]",
    )
    parser.add_argument(
        "modules",
        nargs="+",
        choices=supported_modules,
        help="Modules to run in the experiment pipeline",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Specify what files should be saved.",
    )
    args = parser.parse_args()

    # Initialize configs for each module
    result = []
    new_parser = parser
    for module in args.modules:
        new_parser = create_parser_from_function(
            module_call_parsers[module][1].__call__,
        )
        module_args, unks = new_parser.parse_known_args(args=args.files)
        print(f"Running module: {module} with args: {module_args}")
        print(f"Unrecognized args: {unks}")
        input("Press Enter to continue...")
        # Add subparser arguments to main parser


if __name__ == "__main__":
    main()
