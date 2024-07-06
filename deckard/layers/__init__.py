from .afr import parser as afr_parser
from .afr import main as afr_main
from .attack import parser as attack_parser
from .attack import main as attack_main
from .clean_data import parser as clean_data_parser
from .clean_data import main as clean_data_main
from .compile import parser as compile_parser
from .compile import main as compile_main
from .data import parser as data_parser
from .data import main as data_main
from .experiment import parser as experiment_parser
from .experiment import main as experiment_main
from .generate_grid import parser as generated_grid_parser
from .generate_grid import main as generated_grid_main
from .hydra_test import main as hydra_test_main
from .model import parser as model_parser
from .model import main as model_main
from .optimise import main as optimise_main
from .parse import parser as parse_parser
from .parse import main as parse_main
from .plots import parser as plots_parser
from .plots import main as plots_main
from .prepare_queue import main as prepare_queue_main
from .query_kepler import parser as query_kepler_parser
from .query_kepler import main as query_kepler_main
from .watcher import parser as watcher_parser
from .watcher import main as watcher_main


deckard_layer_dict = {
    "afr": (afr_parser, afr_main),
    "attack": (attack_parser, attack_main),
    "clean_data": (clean_data_parser, clean_data_main),
    "compile": (compile_parser, compile_main),
    "data": (data_parser, data_main),
    "experiment": (experiment_parser, experiment_main),
    "generate_grid": (generated_grid_parser, generated_grid_main),
    "model": (model_parser, model_main),
    "parse": (parse_parser, parse_main),
    "plots": (plots_parser, plots_main),
    "query_kepler": (query_kepler_parser, query_kepler_main),
    "watcher": (watcher_parser, watcher_main),
    "hydra_test": (None, hydra_test_main),
    "optimise": (None, optimise_main),
    "prepare_queue": (None, prepare_queue_main),
}
