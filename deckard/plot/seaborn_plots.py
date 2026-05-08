import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional


from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

# Supported plot types
from seaborn import scatterplot, lineplot, histplot, catplot, barplot, heatmap

from ..utils import load_data, ConfigBase

logger = logging.getLogger(__name__)

seaborn_plotter_dict = {
    "scatter": scatterplot,
    "line": lineplot,
    "hist": histplot,
    "cat": catplot,
    "bar": barplot,
    "heatmap": heatmap,
}

supported_seaborn_plotters = list(seaborn_plotter_dict.keys())


@dataclass(kw_only=True)
class SeabornPlotConfig(ConfigBase):
    """Configuration for seaborn plots"""

    x: str
    y: str
    kwargs: dict = field(default_factory=dict)
    rc_config: dict = field(default_factory=dict)
    plot_type: Literal["scatter", "line", "hist", "cat", "bar", "heatmap"] = "scatter"
    data_file: Optional[str] = None
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xscale: Optional[str] = None
    yscale: Optional[str] = None
    hue: Optional[str] = None
    style: Optional[str] = None
    plot_file: Optional[str] = None
    legend_title: Optional[str] = None
    data: Optional[pd.DataFrame] = None

    def __post_init__(self):
        # Accept either an in-memory DataFrame or a data file path.
        assert (
            self.data is not None or self.data_file is not None
        ), "Either data or data_file must be provided."
        if self.data is not None:
            data = self.data.copy()
        else:
            assert Path(
                self.data_file,
            ).exists(), f"File: {self.data_file} not found."
            data = load_data(self.data_file)
        # Validate columns are in data
        assert (
            self.x in data.columns
        ), f"x value: {self.x} is not a column of the data."
        assert (
            self.y in data.columns
        ), f"y value: {self.y} is not a column of the data."
        if self.hue:
            assert (
                self.hue in data.columns
            ), f"hue value: {self.hue} is not a column of the data."
        if self.style:
            assert (
                self.style in data.columns
            ), f"style value: {self.style} is not a column of the data."
        # Assign data to self.data
        self.data = data

    def __len__(self):
        return 1

    def __call__(self, ax: Optional[Axes] = None):
        plotter_map = globals().get(
            "seaborn_plotter_dict",
            globals().get("searborn_plotter_dict"),
        )
        plotter = plotter_map[self.plot_type]

        if ax is None:
            _, ax = plt.subplots()

        if self.rc_config:
            plt.rcParams.update(self.rc_config)

        plot_kwargs = {
            "data": self.data,
            "x": self.x,
            "y": self.y,
            **self.kwargs,
        }
        if self.hue is not None:
            plot_kwargs["hue"] = self.hue
        if self.style is not None and self.plot_type in ["scatter", "line"]:
            plot_kwargs["style"] = self.style

        try:
            graph = plotter(ax=ax, **plot_kwargs)
        except TypeError:
            graph = plotter(**plot_kwargs)
            if hasattr(graph, "ax"):
                ax = graph.ax

        if self.title:
            ax.set_title(self.title)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)
        if self.xscale:
            ax.set_xscale(self.xscale)
        if self.yscale:
            ax.set_yscale(self.yscale)

        if self.legend_title:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(self.legend_title)
        if self.plot_file:
            plot_path = Path(self.plot_file)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(plot_path, bbox_inches="tight")
        return ax


@dataclass(eq=False)
class SeabornPlotConfigList(ConfigBase):
    plots: List[SeabornPlotConfig] = field(default_factory=list)
    data_file: str = field(default_factory=str)

    def __post_init__(self):
        # Validate self.data_file
        assert Path(
            self.data_file,
        ).exists(), f"File: {self.data_file} not found."

    def __iter__(self):
        return iter(self.plots)

    def __len__(self):
        return len(self.plots)

    def __call__(self, axes=None):
        plot_length = len(self)
        fig = None
        if axes is None:
            fig, axes = plt.subplots(
                nrows=plot_length,
                ncols=1,
                figsize=(10, 8 * plot_length),
            )
        for i in range(plot_length):
            ax = axes[i] if plot_length > 1 else axes
            cfg = self.plots[i]
            try:
                ax = cfg(ax)
            except Exception as e:
                logger.debug(
                    f"Failed to render plot_type: {cfg.plot_type} with error: {e}",
                )
        if fig is not None:
            fig.tight_layout()
        return axes
