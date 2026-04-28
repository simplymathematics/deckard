from .compile_results import compile_results_main, compile_results_parser
from .survival import survival_main, survival_parser
from .optimize import optimize_main, hydra_parser

layer_dict = {
    "compile_results" : [compile_results_parser, compile_results_main], 
    "survival" : [survival_parser, survival_main],
    "optimize" : [hydra_parser, optimize_main],
}
SUPPORTED_LAYERS = list(layer_dict.keys())