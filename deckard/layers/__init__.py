"""CLI layer registry for Deckard subcommands.

The :mod:`deckard.layers` package exposes the parser/main-function pairs used by
the top-level ``deckard`` CLI router.
"""

from .compile_results import compile_results_main, compile_results_parser
from .progress_bar import progress_bar_main, progress_bar_parser
from .plot import plot_main, plot_parser
from .optimize import optimize_main, hydra_parser

try:
    from .survival import survival_main, survival_parser
except ImportError:  # pragma: no cover
    survival_main = None
    survival_parser = None

#: Map CLI subcommand names to ``[parser, main_function]`` pairs consumed by
#: :mod:`deckard.__main__` during subcommand routing.
layer_dict = {
    "compile_results": [compile_results_parser, compile_results_main],
    "progress_bar": [progress_bar_parser, progress_bar_main],
    "plot": [plot_parser, plot_main],
    "optimize": [hydra_parser, optimize_main],
}

if survival_parser is not None and survival_main is not None:
    layer_dict["survival"] = [survival_parser, survival_main]

#: Ordered list of supported CLI layer names derived from :data:`layer_dict`.
SUPPORTED_LAYERS = list(layer_dict.keys())

__all__ = [
    "compile_results_main",
    "compile_results_parser",
    "progress_bar_main",
    "progress_bar_parser",
    "plot_main",
    "plot_parser",
    "optimize_main",
    "hydra_parser",
    "layer_dict",
    "SUPPORTED_LAYERS",
]

if survival_parser is not None and survival_main is not None:
    __all__.extend(["survival_main", "survival_parser"])
