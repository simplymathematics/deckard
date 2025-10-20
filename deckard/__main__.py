import sys
import argparse
import logging
from pathlib import Path

from deckard import data_parser, model_parser, attack_parser
from deckard import data_main, model_main, attack_main

logger = logging.getLogger(__name__)

supported_modules = ["data", "model", "attack"]

# Assert that there is a parser and a main function for each supported module
for module in supported_modules:
    assert hasattr(
        sys.modules[__name__],
        f"{module}_parser",
    ), f"Missing parser for module: {module}"
    assert hasattr(
        sys.modules[__name__],
        f"{module}_main",
    ), f"Missing main function for module: {module}"


def main():
    parser = argparse.ArgumentParser(description="Deckard Command Line Interface")
    parser.add_argument(
        "module",
        choices=["data", "model", "attack", None],
        help="Module to run: data, model, or attack",
    )
    args = parser.parse_known_args()[0]
    
    working_dir = Path(".").resolve()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(working_dir / "deckard.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info(f"Working directory: {working_dir}")
    match args.module:
        case "data":
            args, unknown = data_parser.parse_known_args()
            if len(unknown) > 1:
                logging.error(f"Unknown arguments for data module: {unknown}")
                sys.exit(1)
            data_main(args)
        case "model":
            subparser = argparse.ArgumentParser(
                description="ModelConfig parameters",
                parents=[data_parser, model_parser],
            )
            args, unknown = subparser.parse_known_args()
            if len(unknown) > 1:
                logging.error(f"Unknown arguments for model module: {unknown}")
                sys.exit(1)
            model_main(args)
        case "attack":
            subparser = argparse.ArgumentParser(
                description="AttackConfig parameters",
                parents=[data_parser, model_parser, attack_parser],
            )
            args, unknown = subparser.parse_known_args()
            if len(unknown) > 1:
                logging.error(f"Unknown arguments for attack module: {unknown}")
                sys.exit(1)
            attack_main(args)
        case _:
            parser.print_help()
            logging.error(
                "No valid module specified. Please choose from: data, model, attack.",
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
