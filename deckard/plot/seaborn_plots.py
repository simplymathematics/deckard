from pathlib import Path
import pandas as pd
import json
import yaml
import logging
from dataclasses import dataclass
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from typing import Literal

import seaborn as sns
import matplotlib.pyplot as plt
# import axis object from matplotlib


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from ..utils import ConfigBase
from matplotlib.axes import Axes
logger = logging.getLogger(__name__)


seaborn_supported_plot_types = [
    "line",
    "scatter",
    "cat",
    "heatmap",
    "hist",
]
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
            assert isinstance(ax, )
        if self.plot_type == "line":
            plotter = sns.lineplot
        elif self.plot_type == "scatter":
            plotter = sns.scatterplot
        elif self.plot_type == "heatmap":
            plotter = sns.heatmap
        elif self.plot_type == "hist":
            plotter = sns.histplot
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
        return plot
        
@dataclass
class SeabornCatPlotConfig(ConfigBase):
    row: str = None
    col: str = None
    plot_type: Literal["cat"] = "cat"
    title: str = None
    xlabel: str = None
    ylabel: str = None
    row: str = None
    col: str = None
    hue: str = None
    legend: dict = {}
    x_scale: str = "linear"
    y_scale: str = "linear"
    hue_order: list = None
    row_order: list = None
    col_order: list = None
    file: str = None
    kwargs: dict = {}

    def __call__(self, df: pd.DataFrame, ax = None):
        if ax is None:
            # Set up fig, ax 
            _, ax = plt.subplot(0)
        else:
            assert isinstance(ax, Axes)
        if self.plot_type == "cat":
            plotter = sns.catplot
        else:
            raise NotImplementedError(f"Plot type: {self.plot_type} not supported.")
        g = plotter(
            data=df, 
            x=self.x, 
            y=self.y, 
            hue=self.hue,
            row=self.row,
            col=self.col,
            hue_order=self.hue_order,
            row_order=self.row_order,
            col_order=self.col_order,
            xscale=self.x_scale,
            yscale=self.y_scale,
            **self.kwargs,
            ax = ax,
        )
        if self.title:
            plt.subplots_adjust(top=0.9)
            g.figure.suptitle(self.title)
        if self.xlabel:
            g.set_axis_labels(self.xlabel, self.ylabel)
        if len(self.legend) > 0:
            g.add_legend(**self.legend)
        
        if self.file:
            Path(self.file).parent.mkdir(parents=True, exist_ok=True)
            g.savefig(self.file)
        return g

@dataclass
class SeabornPlotListConfig(ConfigBase):
    plot_configs: list[SeabornPlotConfig]

    def __call__(self, df: pd.DataFrame):
        # Call each plot config on the dataframe, in parallel if possible
        Parallel(n_jobs=-1)(delayed(plot_config)(df) for plot_config in self.plot_configs)