#!/usr/bin/env python3
import argparse
import logging
from omegaconf import OmegaConf
from deckard.layers import deckard_layer_dict

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
    print("Running deckard")
    import sys

    print(sys.argv)
    input("Press Enter to continue...")
    main(args)
