from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import json
import yaml
import logging
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from typing import Literal

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from ..utils import ConfigBase
from matplotlib.axes import Axes
logger = logging.getLogger(__name__)

@dataclass
class SeabornPlotConfig(ConfigBase):
    x : str
    y : str
    plot_type: Literal["line", "scatter"]
    title: str = None
    xlabel: str = None
    ylabel: str = None
    hue: str = None
    legend: dict = {}
    x_scale: str = "linear"
    y_scale: str = "linear"
    hue_order: list = None
    file: str = None
    kwargs: dict = {}


    def __call__(self, df: pd.DataFrame, ax = None):
        if ax is None:
            # Set up fig, ax 
            _, ax = plt.subplot(0)
        else:
            assert isinstance(ax, Axes)
        if self.plot_type == "line":
            plotter = sns.lineplot
        elif self.plot_type == "scatter":
            plotter = sns.scatterplot
        else:
            raise NotImplementedError(f"Plot type: {self.plot_type} not supported.")
        plot = plotter(
            data=df, 
            x=self.x, 
            y=self.y, 
            hue=self.hue,
            hue_order=self.hue_order,
            **self.kwargs,
            ax = ax,
        )
        if self.title:
            plot.set_title(self.title)
        if self.xlabel:
            plot.set_xlabel(self.xlabel)
        if self.ylabel:
            plot.set_ylabel(self.ylabel)
        plot.set_xscale(self.x_scale)
        plot.set_yscale(self.y_scale)
        if len(self.legend) > 0:
            plot.legend(**self.legend)
        
        if self.file:
            Path(self.file).parent.mkdir(parents=True, exist_ok=True)
            plot.get_figure().savefig(self.file)
        