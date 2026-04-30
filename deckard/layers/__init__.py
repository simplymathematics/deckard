from .compile_results import compile_results_main, compile_results_parser
from .progress_bar import progress_bar_main, progress_bar_parser
from .plot import plot_main, plot_parser
from .optimize import optimize_main, hydra_parser

try:
    from .survival import survival_main, survival_parser
except ImportError:  # pragma: no cover
    survival_main = None
    survival_parser = None

layer_dict = {
    "compile_results" : [compile_results_parser, compile_results_main], 
    "progress_bar": [progress_bar_parser, progress_bar_main],
    "plot": [plot_parser, plot_main],
    "optimize" : [hydra_parser, optimize_main],
}

if survival_parser is not None and survival_main is not None:
    layer_dict["survival"] = [survival_parser, survival_main]

SUPPORTED_LAYERS = list(layer_dict.keys())