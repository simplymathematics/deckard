import logging
from dataclasses import dataclass, field
from typing import Union

from ..utils import ConfigBase
from .seaborn_plots import SeabornPlotConfig, SeabornPlotConfigList
from .yellowbrick_plots import YellowBrickConfigList, YellowbrickPlotConfig


logger = logging.getLogger(__name__)


@dataclass
class PlotConfig(ConfigBase):
    kwargs: dict = field(default_factory=dict)
    config: Union[SeabornPlotConfig, SeabornPlotConfigList, YellowbrickPlotConfig, YellowBrickConfigList] = field(init=False, repr=False)

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
            raise ValueError("Missing required source key: provide 'experiment' or 'data_file'.")

        if has_experiment:
            config_cls = YellowBrickConfigList if "plots" in self.kwargs else YellowbrickPlotConfig
        else:
            config_cls = SeabornPlotConfigList if "plots" in self.kwargs else SeabornPlotConfig

        self.config = config_cls(**self.kwargs)

    def __call__(self, *args, **kwargs):
        return self.config(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.config, name)
    
    def __len__(self):
        return len(self.config)
    
    