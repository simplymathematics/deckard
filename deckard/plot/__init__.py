import logging
from dataclasses import dataclass, field
from typing import Union, Optional

from ..utils import ConfigBase
from ..experiment import ExperimentConfig

try:
    from .seaborn_plots import (
        SeabornPlotConfig,
        SeabornPlotConfigList,
        SurvivalSeabornPlotterConfig,
    )
except ImportError:  # pragma: no cover
    SeabornPlotConfig = None
    SeabornPlotConfigList = None
    SurvivalSeabornPlotterConfig = None

try:
    from .yellowbrick_plots import YellowbrickConfigList, YellowbrickPlotConfig
except ImportError:  # pragma: no cover
    YellowbrickConfigList = None
    YellowbrickPlotConfig = None


logger = logging.getLogger(__name__)


@dataclass
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
            raise ValueError("Provide either 'experiment' or 'data_file', not both.")
        if not has_experiment and not has_data_file:
            raise ValueError(
                "Missing required source key: provide 'experiment' or 'data_file'."
            )

        if has_experiment:
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
