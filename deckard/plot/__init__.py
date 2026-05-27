"""Public plotting configuration exports.

``PlotConfig`` is the stable wrapper entrypoint for deckard plotting from Python.
It dispatches to seaborn-backed or yellowbrick-backed concrete plot config
implementations depending on the provided inputs and installed optional
dependencies.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Union

import matplotlib as mpl

from .canon import normalize_plot_backend
from ..utils import BaseConfig

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


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

try:
    from ..plugins.seaborn.plot import (
        SeabornPlotConfig,
        SeabornPlotConfigList,
    )
except ImportError:  # pragma: no cover
    SeabornPlotConfig = None
    SeabornPlotConfigList = None

try:
    from ..plugins.lifelines.plot import (
        SurvivalSeabornPlotConfigList,
        SurvivalSeabornPlotterConfig,
    )

    _ = (SurvivalSeabornPlotConfigList, SurvivalSeabornPlotterConfig)
except ImportError:  # pragma: no cover
    logger.debug(
        "Lifelines not found. Survival plotting configs are unavailable.",
    )
    SurvivalSeabornPlotConfigList = None
    SurvivalSeabornPlotterConfig = None

try:
    from ..plugins.yellowbrick.plot import YellowbrickConfigList, YellowbrickPlotConfig

    _ = (YellowbrickConfigList, YellowbrickPlotConfig)
except ImportError:  # pragma: no cover
    YellowbrickConfigList = None
    YellowbrickPlotConfig = None


def _refresh_seaborn_configs() -> None:
    """Attempt to resolve seaborn plot config classes after module import time."""
    global SeabornPlotConfig, SeabornPlotConfigList
    if SeabornPlotConfig is not None and SeabornPlotConfigList is not None:
        return
    try:
        from ..plugins.seaborn.plot import (
            SeabornPlotConfig as _SeabornPlotConfig,
        )
        from ..plugins.seaborn.plot import (
            SeabornPlotConfigList as _SeabornPlotConfigList,
        )

        if SeabornPlotConfig is None:
            SeabornPlotConfig = _SeabornPlotConfig
        if SeabornPlotConfigList is None:
            SeabornPlotConfigList = _SeabornPlotConfigList
    except ImportError:  # pragma: no cover
        pass


def _refresh_yellowbrick_configs() -> None:
    """Attempt to resolve yellowbrick plot config classes after module import time."""
    global YellowbrickPlotConfig, YellowbrickConfigList
    if YellowbrickPlotConfig is not None and YellowbrickConfigList is not None:
        return
    try:
        from ..plugins.yellowbrick.plot import (
            YellowbrickConfigList as _YellowbrickConfigList,
        )
        from ..plugins.yellowbrick.plot import (
            YellowbrickPlotConfig as _YellowbrickPlotConfig,
        )

        if YellowbrickConfigList is None:
            YellowbrickConfigList = _YellowbrickConfigList
        if YellowbrickPlotConfig is None:
            YellowbrickPlotConfig = _YellowbrickPlotConfig
    except ImportError:  # pragma: no cover
        pass


@dataclass(eq=False, kw_only=True)
class PlotConfig(BaseConfig):
    """Wrapper that routes to appropriate plot config (Seaborn or Yellowbrick).
    
    Takes either an `experiment` (ExperimentConfig) or seaborn data source
    (`data_file`, `data_config`, `data`, or `optuna_storage`) parameter
    to determine the data source, then creates the appropriate plot config.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    kwargs: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    times: dict = field(default_factory=dict)
    plot_state: dict = field(default_factory=dict)
    config: Union[
        SeabornPlotConfig,
        SeabornPlotConfigList,
        YellowbrickPlotConfig,
        YellowbrickConfigList,
    ] = field(init=False, repr=False)

    @staticmethod
    def _resolve_requested_backend(kwargs: dict) -> str | None:
        """Resolve the canonical backend from the supported backend key."""
        raw_backend = kwargs.get("backend")
        if raw_backend is not None:
            return normalize_plot_backend(raw_backend)
        return None

    def __post_init__(self):
        # Merge any extra attributes set by BaseConfig into kwargs
        known_fields = {"kwargs", "files", "times", "plot_state", "config"}
        for attr in list(vars(self).keys()):
            if attr not in known_fields:
                self.kwargs.setdefault(attr, getattr(self, attr))

        has_experiment = self.kwargs.get("experiment") is not None
        has_seaborn_source = any(
            self.kwargs.get(key) is not None
            for key in ("data_file", "data_config", "data", "optuna_storage")
        )
        requested_backend = self._resolve_requested_backend(self.kwargs)

        if has_experiment and has_seaborn_source:
            raise ValueError(
                "Provide either 'experiment' or 'data_file', not both.",
            )
        if not has_experiment and not has_seaborn_source:
            raise ValueError(
                "Missing required source key: provide 'experiment' or 'data_file'.",
            )

        if requested_backend is None:
            backend = "yellowbrick" if has_experiment else "seaborn"
        else:
            backend = requested_backend

        if has_experiment:
            if backend != "yellowbrick":
                raise ValueError(
                    "Experiment source requires yellowbrick backend.",
                )
            _refresh_yellowbrick_configs()
            if YellowbrickPlotConfig is None or YellowbrickConfigList is None:
                raise ImportError(
                    "Yellowbrick plotting requires optional dependency deckard[yellowbrick]",
                )
            config_cls = (
                YellowbrickConfigList
                if "plots" in self.kwargs
                else YellowbrickPlotConfig
            )
        else:
            if backend != "seaborn":
                raise ValueError(
                    "Seaborn data source requires seaborn backend.",
                )
            _refresh_seaborn_configs()
            if SeabornPlotConfig is None or SeabornPlotConfigList is None:
                raise ImportError(
                    "Seaborn plotting requires optional dependency deckard[seaborn]",
                )
            config_cls = (
                SeabornPlotConfigList if "plots" in self.kwargs else SeabornPlotConfig
            )

        self.plot_state["backend"] = backend
        self.plot_state["configured"] = True
        # Keep both keys in sync for legacy and canonical callers.
        self.kwargs["backend"] = backend
        plot_file = self.kwargs.get("plot_file")
        if plot_file is not None:
            self.files["plot_file"] = str(plot_file)

        config_kwargs = dict(self.kwargs)
        config_kwargs.pop("backend", None)
        self.config = config_cls(**config_kwargs)

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
