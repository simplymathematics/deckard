"""Public plotting configuration exports.

``PlotConfig`` is the stable wrapper entrypoint for deckard plotting from Python.
It dispatches to seaborn-backed or yellowbrick-backed concrete plot config
implementations depending on the provided inputs and installed optional
dependencies.
"""

import logging
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

import matplotlib as mpl

from .._optional import load_optional_surface_exports
from .canon import ensure_plot_runtime_contract, normalize_plot_backend
from ..plugins import is_plugin_available
from ..utils import BaseConfig

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_OPTIONAL_PLOT_SURFACE = "deckard.plot"


def _load_default_matplotlibrc() -> None:
    """Load canonical matplotlib rc settings for deckard plotting defaults."""
    rc_path = Path(__file__).resolve().parents[1] / "plots" / ".matplotlibrc"
    if not rc_path.exists():
        logger.debug(
            "No canonical matplotlibrc found at %s; using matplotlib defaults.",
            rc_path,
        )
        return
    try:
        mpl.rc_file(str(rc_path))
        logger.debug("Loaded canonical matplotlibrc from %s", rc_path)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Failed to load canonical matplotlibrc %s: %s",
            rc_path,
            exc,
        )


_load_default_matplotlibrc()

SeabornPlotConfig = None
SeabornPlotConfigList = None
SurvivalSeabornPlotConfigList = None
SurvivalSeabornPlotterConfig = None
YellowbrickPlotConfig = None
YellowbrickConfigList = None

load_optional_surface_exports(
    _OPTIONAL_PLOT_SURFACE,
    module_globals=globals(),
    family="seaborn",
)
load_optional_surface_exports(
    _OPTIONAL_PLOT_SURFACE,
    module_globals=globals(),
    family="lifelines",
)
load_optional_surface_exports(
    _OPTIONAL_PLOT_SURFACE,
    module_globals=globals(),
    family="yellowbrick",
)


def _refresh_seaborn_configs() -> None:
    """Attempt to resolve seaborn plot config classes after module import time."""
    global SeabornPlotConfig, SeabornPlotConfigList
    if SeabornPlotConfig is not None and SeabornPlotConfigList is not None:
        return
    if not is_plugin_available("seaborn"):
        return
    load_optional_surface_exports(
        _OPTIONAL_PLOT_SURFACE,
        module_globals=globals(),
        family="seaborn",
    )


def _refresh_yellowbrick_configs() -> None:
    """Attempt to resolve yellowbrick plot config classes after module import time."""
    global YellowbrickPlotConfig, YellowbrickConfigList
    if YellowbrickPlotConfig is not None and YellowbrickConfigList is not None:
        return
    if not is_plugin_available("yellowbrick"):
        return
    load_optional_surface_exports(
        _OPTIONAL_PLOT_SURFACE,
        module_globals=globals(),
        family="yellowbrick",
    )


@dataclass(eq=False, kw_only=True)
class PlotConfig(BaseConfig):
    """Wrapper that routes to appropriate plot config (Seaborn or Yellowbrick).

    Takes either an `experiment` (ExperimentConfig) or seaborn data source
    (`data_file`, `data_config`, `data`, or `optuna_storage`) parameter
    to determine the data source, then creates the appropriate plot config.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Configuration field: kwargs."},
    )
    files: dict = field(
        default_factory=dict,
        metadata={"help": "Configuration field: files."},
    )
    times: dict = field(
        default_factory=dict,
        metadata={"help": "Configuration field: times."},
    )
    plot_state: dict = field(
        default_factory=dict,
        metadata={"help": "Configuration field: plot_state."},
    )
    config: Union[
        SeabornPlotConfig,
        SeabornPlotConfigList,
        YellowbrickPlotConfig,
        YellowbrickConfigList,
    ] = field(
        init=False,
        repr=False,
        metadata={"help": "Configuration field: config."},
    )

    @staticmethod
    def _resolve_requested_backend(kwargs: dict) -> str | None:
        """Resolve the canonical backend from the supported backend key."""
        raw_backend = kwargs.get("backend")
        if raw_backend is not None:
            return normalize_plot_backend(raw_backend)
        return None

    @staticmethod
    def _filter_backend_kwargs(config_cls, kwargs: dict) -> dict:
        """Keep only constructor-accepted kwargs for backend config classes."""
        signature = inspect.signature(config_cls)
        parameters = signature.parameters

        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return kwargs

        allowed = {
            name
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        return {key: value for key, value in kwargs.items() if key in allowed}

    def _merge_runtime_kwargs(self) -> None:
        known_fields = {"kwargs", "files", "times", "plot_state", "config"}
        for attr in list(vars(self).keys()):
            if attr not in known_fields:
                self.kwargs.setdefault(attr, getattr(self, attr))

    def _resolve_plot_source_flags(self) -> tuple[bool, bool]:
        has_experiment = self.kwargs.get("experiment") is not None
        has_seaborn_source = any(
            self.kwargs.get(key) is not None
            for key in ("data_file", "data_config", "data", "optuna_storage")
        )
        return has_experiment, has_seaborn_source

    @staticmethod
    def _resolve_backend_name(
        requested_backend: str | None,
        has_experiment: bool,
    ) -> str:
        if requested_backend is not None:
            return requested_backend
        return "yellowbrick" if has_experiment else "seaborn"

    @staticmethod
    def _resolve_yellowbrick_config_class(wants_list: bool):
        if wants_list:
            if YellowbrickConfigList is None:
                _refresh_yellowbrick_configs()
            if YellowbrickConfigList is None:
                raise ImportError(
                    "Yellowbrick plotting requires optional dependency deckard[yellowbrick]",
                )
            return YellowbrickConfigList

        if YellowbrickPlotConfig is None:
            _refresh_yellowbrick_configs()
        if YellowbrickPlotConfig is None:
            raise ImportError(
                "Yellowbrick plotting requires optional dependency deckard[yellowbrick]",
            )
        return YellowbrickPlotConfig

    @staticmethod
    def _resolve_seaborn_config_class(wants_list: bool):
        if wants_list:
            if SeabornPlotConfigList is None:
                _refresh_seaborn_configs()
            if SeabornPlotConfigList is None:
                raise ImportError(
                    "Seaborn plotting requires optional dependency deckard[seaborn]",
                )
            return SeabornPlotConfigList

        if SeabornPlotConfig is None:
            _refresh_seaborn_configs()
        if SeabornPlotConfig is None:
            raise ImportError(
                "Seaborn plotting requires optional dependency deckard[seaborn]",
            )
        return SeabornPlotConfig

    def _resolve_plot_config_class(self, backend: str, has_experiment: bool):
        wants_list = "plots" in self.kwargs
        if has_experiment:
            if backend != "yellowbrick":
                raise ValueError("Experiment source requires yellowbrick backend.")
            return self._resolve_yellowbrick_config_class(wants_list)
        if backend != "seaborn":
            raise ValueError("Seaborn data source requires seaborn backend.")
        return self._resolve_seaborn_config_class(wants_list)

    def __post_init__(self):
        ensure_plot_runtime_contract(self)
        self._merge_runtime_kwargs()

        has_experiment, has_seaborn_source = self._resolve_plot_source_flags()
        requested_backend = self._resolve_requested_backend(self.kwargs)

        if has_experiment and has_seaborn_source:
            raise ValueError(
                "Provide either 'experiment' or 'data_file', not both.",
            )
        if not has_experiment and not has_seaborn_source:
            raise ValueError(
                "Missing required source key: provide 'experiment' or 'data_file'.",
            )

        backend = self._resolve_backend_name(requested_backend, has_experiment)
        config_cls = self._resolve_plot_config_class(backend, has_experiment)

        self.plot_state["backend"] = backend
        self.plot_state["configured"] = True
        # Keep both keys in sync for legacy and canonical callers.
        self.kwargs["backend"] = backend
        plot_file = self.kwargs.get("plot_file")
        if plot_file is not None:
            self.files["plot_file"] = str(plot_file)

        config_kwargs = dict(self.kwargs)
        config_kwargs.pop("backend", None)
        self.config = config_cls(
            **self._filter_backend_kwargs(config_cls, config_kwargs),
        )

    def __call__(self, *args, **kwargs) -> Union[dict, "Axes"]:
        """Render the resolved plotting backend and return its runtime output.

        Args:
            *args: Positional args forwarded to resolved plot config.
            **kwargs: Keyword args forwarded to resolved plot config.

        Returns:
            Plot backend runtime output payload.
        """
        start = time.perf_counter()
        out = self.config(*args, **kwargs)
        self.times["plot_call_time"] = time.perf_counter() - start
        self.plot_state["rendered"] = True
        return out

    def __getattr__(self, name):
        return getattr(self.config, name)

    def __len__(self):
        return len(self.config)


__all__ = [
    "PlotConfig",
    "SeabornPlotConfig",
    "SeabornPlotConfigList",
    "SurvivalSeabornPlotConfigList",
    "SurvivalSeabornPlotterConfig",
    "YellowbrickConfigList",
    "YellowbrickPlotConfig",
]
