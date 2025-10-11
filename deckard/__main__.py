import sys
from deckard import data_parser, model_parser, attack_parser
from deckard import data_main, model_main, attack_main
import argparse
import logging

logger = logging.getLogger(__name__)

supported_modules = ["data", "model", "attack"]

# Assert that there is a parser and a main function for each supported module
for module in supported_modules:
    assert hasattr(sys.modules[__name__], f"{module}_parser"), f"Missing parser for module: {module}"
    assert hasattr(sys.modules[__name__], f"{module}_main"), f"Missing main function for module: {module}"




def main():
    parser = argparse.ArgumentParser(description="Deckard Command Line Interface", parents=[data_parser, model_parser, attack_parser])
    parser.add_argument(
        "module",
        choices=["data", "model", "attack", None],
        help="Module to run: data, model, or attack",
        
    )
    args = parser.parse_args()
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
            logging.error("No valid module specified. Please choose from: data, model, attack.")
            sys.exit(1)

if __name__ == "__main__":
    main()