#!/usr/bin/env python3
import sys
import logging
from omegaconf import OmegaConf
from .layers.afr import aft_parser, aft_main
from .layers.attack import attack_parser, attack_main
from .layers.clean_data import clean_data_parser, clean_data_main
from .layers.compile import compile_parser, compile_main
from .layers.data import data_parser, data_main
from .layers.experiment import experiment_parser, experiment_main
from .layers.find_best import find_best_parser, find_best_main
from .layers.generate_grid import generate_grid_parser, generate_grid_main
from .layers.hydra_test import hydra_test_main
from .layers.merge import merge_parser, merge_main
from .layers.optimise import optimise_main
from .layers.parse import hydra_parser, parse_hydra_config
from .layers.plots import plots_parser, plots_main
from .layers.prepare_queue import prepare_queue_main
from .layers.query_kepler import kepler_parser, kepler_main

OmegaConf.register_new_resolver("eval", eval)

logger = logging.getLogger(__name__)
layer_list = [
    "afr",
    "attack",
    "clean_data" "compile",
    "data",
    "experiment",
    "find_best",
    "generate_grid",
    "hydra_test",
    "merge",
    "optimise",
    "parse",
    "plots",
    "prepare_queue",
    "query_kepler",
]


deckard_layer_dict = {
    "afr": (aft_parser, aft_main),
    "attack": (attack_parser, attack_main),
    "clean_data": (clean_data_parser, clean_data_main),
    "compile": (compile_parser, compile_main),
    "data": (data_parser, data_main),
    "experiment": (experiment_parser, experiment_main),
    "find_best": (find_best_parser, find_best_main),
    "generate_grid": (generate_grid_parser, generate_grid_main),
    "hydra_test": (None, hydra_test_main),
    "merge": (merge_parser, merge_main),
    "optimise": (None, optimise_main),
    "parse": (hydra_parser, parse_hydra_config),
    "plots": (plots_parser, plots_main),
    "prepare_queue": (None, prepare_queue_main),
    "query_kepler": (kepler_parser, kepler_main),
}
assert len(deckard_layer_dict) == len(
    layer_list,
), "Some layers are missing from the deckard_layer_dict"


def main(layer, args):
    # Get the layer and the main function for the layer.
    if layer not in deckard_layer_dict:
        raise ValueError(f"Layer {layer} not found.")
    parser, sub_main = deckard_layer_dict[layer]
    # Parse the arguments.
    args = parser.parse_args(args.args)
    # Print the arguments and values
    # Run the main function.
    sub_main(args)


if __name__ == "__main__":
    # pop the first argument which is the script name
    layer = sys.argv.pop(1)
    # pass the rest of the arguments to the main function
    main(layer, sys.argv)
