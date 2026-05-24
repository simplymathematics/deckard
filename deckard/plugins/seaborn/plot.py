import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

# Supported plot types
from seaborn import barplot, catplot, heatmap, histplot, lineplot, scatterplot

from ...plot.base import (
    PlotterMixin,
    _SeabornPlotterMarker,
)
from ...optuna_callback import load_optuna_studies_dataframe
from ...utils import ConfigBase, load_data

if TYPE_CHECKING:
    from ...data import DataConfig

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


@dataclass(eq=True)
class SeabornPlotterMixin(PlotterMixin):
    """Seaborn-specific plotter handler for matplotlib-based rendering.

    Initialization parameters
    -------------------------
    runtime : Any
        Seaborn plot config object (SeabornPlotConfig or subclass).

    Runtime parameters
    -------------------
    plot_type : str
        Seaborn plot type ("scatter", "line", "hist", "cat", "bar", "heatmap").
    data : pd.DataFrame
        Data for plotting, materialized from file or passed directly.
    x, y : str
        Column names for x and y axes.
    hue, style : str | None
        Optional aesthetic mappings.
    rc_config : dict
        Matplotlib rcParams updates.
    kwargs : dict
        Additional plotter-specific parameters.

    Plugin pattern
    --------------
    This mixin is registered via PlotTypePlugin for plot_backend="seaborn"
    and provides seaborn-specific rendering logic when bound to SeabornPlotConfig.
    """

    def __call__(
        self,
        *,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Execute seaborn plot rendering.

        Parameters
        ----------
        ax : Axes | None
            Matplotlib axis to plot on. If None, creates new figure/axis.
        **kwargs : Any
            Additional plot parameters forwarded to seaborn function.

        Returns
        -------
        Axes
            Matplotlib axis containing rendered plot.
        """
        plotter_map = globals().get(
            "seaborn_plotter_dict",
            globals().get("searborn_plotter_dict"),
        )
        plotter = plotter_map[self.runtime.plot_type]

        if ax is None and self.runtime.plot_type != "cat":
            _, ax = plt.subplots()

        if self.runtime.rc_config:
            plt.rcParams.update(self.runtime.rc_config)

        if self.runtime.plot_type == "heatmap":
            plot_kwargs = {
                "data": self.runtime.data,
                **self.runtime.kwargs,
                **kwargs,
            }
        else:
            plot_kwargs = {
                "data": self.runtime.data,
                "x": self.runtime.x,
                "y": self.runtime.y,
                **self.runtime.kwargs,
                **kwargs,
            }
            if self.runtime.hue is not None:
                plot_kwargs["hue"] = self.runtime.hue
            if self.runtime.style is not None and self.runtime.plot_type in [
                "scatter",
                "line",
            ]:
                plot_kwargs["style"] = self.runtime.style

        if self.runtime.plot_type == "cat":
            graph = plotter(**plot_kwargs)
            if hasattr(graph, "ax"):
                ax = graph.ax
            elif hasattr(graph, "axes"):
                axes = graph.axes
                if axes is not None:
                    ax = axes.flat[0] if hasattr(axes, "flat") else axes[0]
            elif hasattr(graph, "figure") and graph.figure.axes:
                ax = graph.figure.axes[0]
        else:
            try:
                graph = plotter(ax=ax, **plot_kwargs)
            except TypeError:
                graph = plotter(**plot_kwargs)
                if hasattr(graph, "ax"):
                    ax = graph.ax
                elif hasattr(graph, "axes"):
                    axes = graph.axes
                    if axes is not None:
                        ax = axes.flat[0] if hasattr(axes, "flat") else axes[0]
                elif hasattr(graph, "figure") and graph.figure.axes:
                    ax = graph.figure.axes[0]

        if self.runtime.title:
            ax.set_title(self.runtime.title)
        if self.runtime.xlabel:
            ax.set_xlabel(self.runtime.xlabel)
        if self.runtime.ylabel:
            ax.set_ylabel(self.runtime.ylabel)
        if self.runtime.xscale:
            ax.set_xscale(self.runtime.xscale)
        if self.runtime.yscale:
            ax.set_yscale(self.runtime.yscale)

        if self.runtime.legend_title:
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(self.runtime.legend_title)
        if self.runtime.plot_file:
            plot_path = Path(self.runtime.plot_file)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.savefig(plot_path, bbox_inches="tight")
        return ax


@dataclass(kw_only=True, eq=False)
class SeabornPlotConfig(_SeabornPlotterMarker, ConfigBase):
    """Configuration for seaborn matplotlib-based plots.

    Initialization parameters
    -------------------------
    x : str
        Column name for x-axis data.
    y : str
        Column name for y-axis data.
    plot_type : Literal["scatter", "line", "hist", "cat", "bar", "heatmap"]
        Type of seaborn plot to render.
    data_file : str | None
        Path to CSV/parquet data file. Mutually exclusive with data parameter.
    data : pd.DataFrame | None
        In-memory data. Mutually exclusive with data_file parameter.
    title : str | None
        Plot title.
    xlabel : str | None
        X-axis label.
    ylabel : str | None
        Y-axis label.
    xscale : str | None
        X-axis scale ("linear", "log", etc).
    yscale : str | None
        Y-axis scale.
    hue : str | None
        Column for color encoding (supports scatter, line).
    style : str | None
        Column for marker/line style (supports scatter, line).
    plot_file : str | None
        Output path for rendered plot.
    legend_title : str | None
        Legend title override.
    kwargs : dict
        Additional seaborn plotter parameters (e.g., s=100 for scatter).
    rc_config : dict
        Matplotlib rcParams updates (e.g., figsize, font).

    Runtime parameters
    -------------------
    None (all parameters determined at initialization).

    Parameter layers
    ----------------
    1. Data source: Either file-based (data_file) or in-memory (data)
    2. Aesthetic encoding: x/y columns plus optional hue/style mappings
    3. Styling: rc_config for matplotlib, kwargs for plotter-specific options
    4. Rendering: Axis customization (scales, labels), file output

    Family-specific parameter semantics
    -----------------------------------
    Seaborn plots provide publication-quality statistical graphics:

    - **scatter**: Bivariate relationships with optional grouping
    - **line**: Time-series or categorical line plots
    - **hist**: Univariate or bivariate distributions
    - **cat**: Categorical plots (box, violin, strip, etc)
    - **bar**: Categorical bar plots with aggregation
    - **heatmap**: Rectangular heatmap with color encoding

    Plugin pattern
    --------------
    This config inherits from ``_SeabornPlotterMarker`` for backend identification.
    At runtime, ``PlotTypePlugin`` resolves ``_SeabornPlotterMixin`` for rendering
    when plot_backend="seaborn", enabling flexible seaborn/yellowbrick switching.
    """

    x: str
    y: str
    kwargs: dict = field(default_factory=dict)
    rc_config: dict = field(default_factory=dict)
    plot_type: Literal["scatter", "line", "hist", "cat", "bar", "heatmap"] = "scatter"
    data_file: Optional[str] = None
    data_config: Optional["DataConfig"] = None
    optuna_storage: Optional[str] = None
    optuna_study_name: Optional[str] = None
    optuna_schema: Optional[Union[dict[str, Any], str]] = None
    optuna_query: dict[str, Any] = field(default_factory=dict)
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
        # Seaborn plotters are an extension of DataConfig runtime payloads.
        # Accept in-memory data, data_file, DataConfig, or optuna-backed source.
        if self.data_config is not None:
            data_obj = self.data_config
            if callable(getattr(data_obj, "__call__", None)) and (
                getattr(data_obj, "_X", None) is None or getattr(data_obj, "_y", None) is None
            ):
                data_obj(files={"data_file": None, "score_file": None})
            data = getattr(data_obj, "_X", None)
            if data is None:
                raise ValueError("Provided data_config did not materialize feature dataframe (_X).")
        elif self.data is not None:
            data = self.data.copy()
        elif self.data_file is not None:
            assert Path(
                self.data_file,
            ).exists(), f"File: {self.data_file} not found."
            data = load_data(self.data_file)
        else:
            storage = self.optuna_storage
            if storage is None and Path("optuna.db").exists():
                storage = "sqlite:///optuna.db"
            assert (
                storage is not None
            ), "Provide one of data, data_file, data_config, or optuna_storage (or optuna.db in cwd)."
            data = load_optuna_studies_dataframe(
                storage=storage,
                study_name=self.optuna_study_name,
                schema=self.optuna_schema,
                **dict(self.optuna_query or {}),
            )

        if self.data is not None:
            data = pd.DataFrame(data).copy()
        else:
            data = pd.DataFrame(data)
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

        if ax is None and self.plot_type != "cat":
            _, ax = plt.subplots()

        if self.rc_config:
            plt.rcParams.update(self.rc_config)

        if self.plot_type == "heatmap":
            plot_kwargs = {
                "data": self.data,
                **self.kwargs,
            }
        else:
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

        if self.plot_type == "cat":
            graph = plotter(**plot_kwargs)
            if hasattr(graph, "ax"):
                ax = graph.ax
            elif hasattr(graph, "axes"):
                axes = graph.axes
                if axes is not None:
                    ax = axes.flat[0] if hasattr(axes, "flat") else axes[0]
            elif hasattr(graph, "figure") and graph.figure.axes:
                ax = graph.figure.axes[0]
        else:
            try:
                graph = plotter(ax=ax, **plot_kwargs)
            except TypeError:
                graph = plotter(**plot_kwargs)
                if hasattr(graph, "ax"):
                    ax = graph.ax
                elif hasattr(graph, "axes"):
                    axes = graph.axes
                    if axes is not None:
                        ax = axes.flat[0] if hasattr(axes, "flat") else axes[0]
                elif hasattr(graph, "figure") and graph.figure.axes:
                    ax = graph.figure.axes[0]

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


@dataclass(eq=False, kw_only=True)
class SeabornPlotConfigList(ConfigBase):
    """Container for multiple seaborn plot configurations.

    Initialization parameters
    -------------------------
    plots : list[SeabornPlotConfig]
        List of plot configurations to render.
    data_file : str
        Shared data file path for all plots (used for validation).

    Runtime parameters
    -------------------
    axes : Axes | None
        Matplotlib axes array to render on. If None, creates new figure/subplots.

    Parameter layers
    ----------------
    1. Plot configurations: List of SeabornPlotConfig instances
    2. Data source: Single shared data_file for all plots
    3. Figure layout: Subplot grid matching number of plots

    Family-specific parameter semantics
    -----------------------------------
    Batch rendering of multiple seaborn plots with automatic subplot arrangement.
    Each plot operates independently on shared or separate data sources.

    Plugin pattern
    --------------
    This container orchestrates multiple ``SeabornPlotConfig`` instances,
    each of which participates in ``PlotTypePlugin`` resolution for backend dispatch.
    """

    plots: List[SeabornPlotConfig] = field(default_factory=list)
    data_file: Optional[str] = None
    data_config: Optional["DataConfig"] = None
    optuna_storage: Optional[str] = None
    optuna_study_name: Optional[str] = None
    optuna_schema: Optional[Union[dict[str, Any], str]] = None

    def __post_init__(self):
        # Keep list containers aligned with SeabornPlotConfig data-source options.
        if self.data_file is not None:
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
