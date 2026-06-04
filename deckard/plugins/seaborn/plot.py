import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union, cast

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
from ...utils import BaseConfig, load_data

if TYPE_CHECKING:
    from ...data import DataConfig

logger = logging.getLogger(__name__)

SeabornScalar = str | int | float | bool | None
SeabornValue = SeabornScalar | list["SeabornValue"] | dict[str, "SeabornValue"]

seaborn_plotter_dict = {
    "scatter": scatterplot,
    "line": lineplot,
    "hist": histplot,
    "cat": catplot,
    "bar": barplot,
    "heatmap": heatmap,
}

supported_seaborn_plotters = list(seaborn_plotter_dict.keys())


def _resolve_seaborn_plotter(plot_type: str):
    plotter_map = globals().get(
        "seaborn_plotter_dict",
        globals().get("searborn_plotter_dict"),
    )
    return plotter_map[plot_type]


def _ensure_plot_axes(ax: Axes | None, plot_type: str) -> Axes | None:
    if ax is None and plot_type != "cat":
        _, ax = plt.subplots()
    return ax


def _build_plot_kwargs(
    *,
    plot_type: str,
    data: Any,
    base_kwargs: dict[str, Any],
    extra_kwargs: dict[str, SeabornValue],
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    style: str | None = None,
) -> dict[str, Any]:
    if plot_type == "heatmap":
        return {
            "data": data,
            **base_kwargs,
            **extra_kwargs,
        }
    plot_kwargs: dict[str, Any] = {
        "data": data,
        "x": x,
        "y": y,
        **base_kwargs,
        **extra_kwargs,
    }
    if hue is not None:
        plot_kwargs["hue"] = hue
    if style is not None and plot_type in ["scatter", "line"]:
        plot_kwargs["style"] = style
    return plot_kwargs


def _axes_from_plot_graph(graph: Any, fallback: Axes | None) -> Axes | None:
    if isinstance(graph, Axes):
        return graph
    if hasattr(graph, "ax"):
        return cast(Axes | None, graph.ax)
    if hasattr(graph, "axes"):
        axes = graph.axes
        if axes is not None:
            return cast(Axes, axes.flat[0] if hasattr(axes, "flat") else axes[0])
    if hasattr(graph, "figure") and graph.figure.axes:
        return cast(Axes, graph.figure.axes[0])
    return fallback


def _render_plot(
    *,
    plotter: Any,
    plot_type: str,
    ax: Axes | None,
    plot_kwargs: dict[str, Any],
) -> Axes | None:
    if plot_type == "cat":
        return _axes_from_plot_graph(plotter(**plot_kwargs), ax)
    try:
        graph = plotter(ax=ax, **plot_kwargs)
    except TypeError:
        graph = plotter(**plot_kwargs)
    return _axes_from_plot_graph(graph, ax)


def _finalize_plot_axes(
    *,
    ax: Axes,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    xscale: str | None,
    yscale: str | None,
    legend_title: str | None,
    plot_file: str | None,
) -> Axes:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)
    if legend_title:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(legend_title)
    if plot_file:
        plot_path = Path(plot_file)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(plot_path, bbox_inches="tight")
    return ax


@dataclass(eq=True)
class SeabornPlotterMixin(PlotterMixin):
    """Seaborn-specific plotter handler for matplotlib-based rendering.

    The runtime object provides the active seaborn plot config, including plot
    type, data payload, aesthetic mappings, rcParams overrides, and output
    paths used during rendering.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def __call__(
        self,
        *,
        ax: Axes | None = None,
        **kwargs: SeabornValue,
    ) -> Axes:
        """Execute seaborn plot rendering.

        Args:
            ax: Matplotlib axis to plot on. If None, creates a new axis.
            **kwargs: Additional plot parameters forwarded to seaborn function.

        Returns:
            Matplotlib axis containing rendered plot.
        """
        plotter = _resolve_seaborn_plotter(self.runtime.plot_type)
        ax = _ensure_plot_axes(ax, self.runtime.plot_type)

        if self.runtime.rc_config:
            plt.rcParams.update(self.runtime.rc_config)

        plot_kwargs = _build_plot_kwargs(
            plot_type=self.runtime.plot_type,
            data=self.runtime.data,
            base_kwargs=dict(self.runtime.kwargs),
            extra_kwargs=kwargs,
            x=self.runtime.x,
            y=self.runtime.y,
            hue=self.runtime.hue,
            style=self.runtime.style,
        )
        resolved_ax = _render_plot(
            plotter=plotter,
            plot_type=self.runtime.plot_type,
            ax=ax,
            plot_kwargs=plot_kwargs,
        )
        if resolved_ax is None:
            _, resolved_ax = plt.subplots()
        return _finalize_plot_axes(
            ax=resolved_ax,
            title=self.runtime.title,
            xlabel=self.runtime.xlabel,
            ylabel=self.runtime.ylabel,
            xscale=self.runtime.xscale,
            yscale=self.runtime.yscale,
            legend_title=self.runtime.legend_title,
            plot_file=self.runtime.plot_file,
        )


@dataclass(kw_only=True, eq=False)
class SeabornPlotConfig(_SeabornPlotterMarker, BaseConfig):
    """Configuration for seaborn matplotlib-based plots.

    This config stores the seaborn plot type, data source, aesthetic channel
    mappings, styling options, and output path used for one rendered plot. It
    inherits ``_SeabornPlotterMarker`` so runtime dispatch resolves the seaborn
    plotting mixin when the seaborn backend is selected.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    x: str
    y: str
    kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Configuration field: kwargs."},
    )
    rc_config: dict = field(
        default_factory=dict,
        metadata={"help": "Configuration field: rc_config."},
    )
    plot_type: Literal["scatter", "line", "hist", "cat", "bar", "heatmap"] = "scatter"
    data_file: Optional[str] = None
    data_config: Optional["DataConfig"] = None
    optuna_storage: Optional[str] = None
    optuna_study_name: Optional[str] = None
    optuna_schema: Optional[Union[dict[str, Any], str]] = None
    optuna_query: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: optuna_query."},
    )
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
                getattr(data_obj, "_X", None) is None
                or getattr(data_obj, "_y", None) is None
            ):
                data_obj(files={"data_file": None, "score_file": None})
            data = getattr(data_obj, "_X", None)
            if data is None:
                raise ValueError(
                    "Provided data_config did not materialize feature dataframe (_X).",
                )
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

    def __call__(self, ax: Optional[Axes] = None) -> Axes:
        """Render the configured seaborn plot and return the resolved axes.

        Args:
            ax: Optional pre-created axis to render into.

        Returns:
            Axis containing the rendered seaborn chart.
        """
        plotter = _resolve_seaborn_plotter(self.plot_type)
        ax = _ensure_plot_axes(ax, self.plot_type)

        if self.rc_config:
            plt.rcParams.update(self.rc_config)

        plot_kwargs = _build_plot_kwargs(
            plot_type=self.plot_type,
            data=self.data,
            base_kwargs=dict(self.kwargs),
            extra_kwargs={},
            x=self.x,
            y=self.y,
            hue=self.hue,
            style=self.style,
        )
        resolved_ax = _render_plot(
            plotter=plotter,
            plot_type=self.plot_type,
            ax=ax,
            plot_kwargs=plot_kwargs,
        )
        if resolved_ax is None:
            _, resolved_ax = plt.subplots()
        return _finalize_plot_axes(
            ax=resolved_ax,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            xscale=self.xscale,
            yscale=self.yscale,
            legend_title=self.legend_title,
            plot_file=self.plot_file,
        )


@dataclass(eq=False, kw_only=True)
class SeabornPlotConfigList(BaseConfig):
    """Container for multiple seaborn plot configurations.

    This config orchestrates batch rendering of multiple ``SeabornPlotConfig``
    instances, including shared data-file validation and subplot layout.
    """

    plots: List[SeabornPlotConfig] = field(
        default_factory=list,
        metadata={"help": "Configuration field: plots."},
    )
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

    def __call__(self, axes: Axes | list[Axes] | None = None) -> Axes | list[Axes]:
        """Render all configured plots and return the resulting axes collection.

        Args:
            axes: Optional axes array for plot placement.

        Returns:
            Axes collection used for rendered plots.
        """
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
