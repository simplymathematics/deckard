"""Public plotting configuration exports.

``PlotConfig`` is the stable wrapper entrypoint for deckard plotting from Python.
It dispatches to seaborn-backed or yellowbrick-backed concrete plot config
implementations depending on the provided inputs and installed optional
dependencies.
"""

import logging
from dataclasses import dataclass, field
from typing import Union

from ..utils import ConfigBase

logger = logging.getLogger(__name__)

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
class PlotConfig(ConfigBase):
    """Wrapper that routes to appropriate plot config (Seaborn or Yellowbrick).

    Takes either an `experiment` (ExperimentConfig) or `data_file` parameter
    to determine the data source, then creates the appropriate plot config.
    """

    kwargs: dict = field(default_factory=dict)
    config: Union[
        SeabornPlotConfig,
        SeabornPlotConfigList,
        YellowbrickPlotConfig,
        YellowbrickConfigList,
    ] = field(init=False, repr=False)

    def __post_init__(self):
        # Merge any extra attributes set by ConfigBase into kwargs
        known_fields = {"kwargs", "config"}
        for attr in list(vars(self).keys()):
            if attr not in known_fields:
                self.kwargs.setdefault(attr, getattr(self, attr))

        has_experiment = self.kwargs.get("experiment") is not None
        has_data_file = self.kwargs.get("data_file") is not None

        if has_experiment and has_data_file:
            raise ValueError(
                "Provide either 'experiment' or 'data_file', not both.",
            )
        if not has_experiment and not has_data_file:
            raise ValueError(
                "Missing required source key: provide 'experiment' or 'data_file'.",
            )

        if has_experiment:
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
            _refresh_seaborn_configs()
            if SeabornPlotConfig is None or SeabornPlotConfigList is None:
                raise ImportError(
                    "Seaborn plotting requires optional dependency deckard[seaborn]",
                )
            config_cls = (
                SeabornPlotConfigList if "plots" in self.kwargs else SeabornPlotConfig
            )

        self.config = config_cls(**self.kwargs)

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)

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
