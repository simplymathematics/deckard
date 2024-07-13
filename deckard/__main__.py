#!/usr/bin/env python3
import argparse
import subprocess
import logging
from pathlib import Path
from omegaconf import OmegaConf
from .layers.parse import save_params_file

OmegaConf.register_new_resolver("eval", eval)

logger = logging.getLogger(__name__)


def main(args):
    # Get the layer and the main function for the layer.
    layer = args.layer
    if layer not in deckard_layer_dict:
        raise ValueError(f"Layer {layer} not found.")
    print("Trying to run layer", layer)
    parser, sub_main = deckard_layer_dict[layer]
    # Parse the arguments.
    args = parser.parse_args(args.args)
    # Print the arguments and values
    import yaml

    print(yaml.dump(OmegaConf.to_container(args)))
    input("Press Enter to continue...")
    # Run the main function.
    sub_main(args)


parser = argparse.ArgumentParser()
# Choose which layers to run.
parser.add_argument("layer", help="The layers to run.")
# The rest of the arguments are passed to the layer.
parser.add_argument(
    "args",
    nargs=argparse.REMAINDER,
    help="Arguments to pass to the layer.",
)
# parse the layer to know which subparser to use.
args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submodule",
        type=str,
        help=f"Submodule to run. Choices: {layer_list}",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="default hydra configuration file that you would like to reproduce with dvc repro.",
    )
    parser.add_argument("--config_dir", type=str, default="conf")
    parser.add_argument("other_args", type=str, nargs="*")
    args = parser.parse_args()
    submodule = args.submodule
    if submodule is not None:
        assert (
            args.config_file is None
        ), "config_file and submodule cannot be specified at the same time"
    if submodule not in layer_list and submodule is not None:
        raise ValueError(f"Submodule {submodule} not found. Choices: {layer_list}")
    if len(args.other_args) > 0:
        other_args = " ".join(args.other_args)
    else:
        other_args = []
    if submodule is None:
        assert (
            parse_and_repro(other_args, args.config_file, config_dir=args.config_dir)
            == 0
        )
    else:
        assert run_submodule(submodule, other_args) == 0
