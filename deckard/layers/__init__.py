"""CLI layer registry for deckard subcommands.

The :mod:`deckard.layers` package exposes the parser/main-function pairs used by
the top-level ``deckard`` CLI router.
"""

from collections.abc import Callable
from typing import TypeAlias

from .compile_results import compile_results_main, compile_results_parser
from .extensions_cli import (
    frameworks_main,
    frameworks_parser,
    plugins_main,
    plugins_parser,
)
from .optimize import hydra_parser, optimize_main
from .plot import plot_main, plot_parser
from .progress_bar import progress_bar_main, progress_bar_parser
from .rerun_failed_studies import (
    rerun_failed_studies_main,
    rerun_failed_studies_parser,
)

try:
    from .survival import survival_main, survival_parser
except ImportError:  # pragma: no cover
    survival_main = None
    survival_parser = None

LayerParser: TypeAlias = Callable[..., object]
LayerMain: TypeAlias = Callable[..., object]

#: Mapping from CLI subcommand name to ``[parser, main]`` callables.
layer_dict: dict[str, list[Callable[..., object]]] = {
    "compile_results": [compile_results_parser, compile_results_main],
    "progress_bar": [progress_bar_parser, progress_bar_main],
    "plot": [plot_parser, plot_main],
    "optimize": [hydra_parser, optimize_main],
    "plugins": [plugins_parser, plugins_main],
    "frameworks": [frameworks_parser, frameworks_main],
    "rerun_failed_studies": [
        rerun_failed_studies_parser,
        rerun_failed_studies_main,
    ],
}

if survival_parser is not None and survival_main is not None:
    layer_dict["survival"] = [survival_parser, survival_main]

#: Ordered list of supported CLI layer names derived from :data:`layer_dict`.
SUPPORTED_LAYERS: list[str] = list(layer_dict.keys())

__all__ = [
    "compile_results_main",
    "compile_results_parser",
    "progress_bar_main",
    "progress_bar_parser",
    "plot_main",
    "plot_parser",
    "optimize_main",
    "hydra_parser",
    "plugins_main",
    "plugins_parser",
    "frameworks_main",
    "frameworks_parser",
    "rerun_failed_studies_main",
    "rerun_failed_studies_parser",
    "layer_dict",
    "SUPPORTED_LAYERS",
]

if survival_parser is not None and survival_main is not None:
    __all__.extend(["survival_main", "survival_parser"])
