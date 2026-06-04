"""Core plotting primitives and default plotter profiles.

This module provides the _Mixin -> Plugin -> Config pattern for plotting,
enabling flexible composition of plot configurations with experiment state.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..utils import BaseConfig
from .canon import normalize_plot_backend

logger = logging.getLogger(__name__)

PlotScalar = str | int | float | bool | None
PlotValue = PlotScalar | list["PlotValue"] | dict[str, "PlotValue"]
PlotResult = dict[str, PlotValue] | BaseConfig | None


class _SeabornPlotterMarker:
    """Mixin that marks a plot config as operating with seaborn backend.

    Inherit this class alongside a plot config to signal seaborn-specific
    plotting behavior and rendering parameters.
    """


class _YellowbrickPlotterMarker:
    """Mixin that marks a plot config as operating with yellowbrick backend.

    Inherit this class alongside a plot config to signal yellowbrick-specific
    visualization and model integration behavior.
    """


@dataclass(eq=True)
class PlotterMixin:
    """Base callable plotter handler used by runtime plotter context resolution.

    The ``runtime`` attribute is the active plot config instance owned by the
    plot config ``__call__`` path. Attribute access is delegated to that
    runtime object so mixins can share experiment state, cached results, and
    plot parameters.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    runtime: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "runtime":
            object.__setattr__(self, name, value)
            return
        runtime = object.__getattribute__(self, "runtime")
        if runtime is None:
            object.__setattr__(self, name, value)
            return
        setattr(runtime, name, value)

    def __call__(
        self,
        *,
        experiment: BaseConfig | None = None,
        plot_type: str = "scatter",
        **kwargs: PlotValue,
    ) -> PlotResult:
        """Execute plotter handler.

        Args:
            experiment: Runtime experiment context.
            plot_type: Type of plot to render.
            **kwargs: Additional plotter-specific parameters.

        Returns:
            Rendered plot payload from concrete plotter implementation.

        Raises:
            NotImplementedError: Always raised by the base mixin implementation.
        """
        raise NotImplementedError(
            "Plotter mixins must implement __call__",
        )


@dataclass(eq=False, kw_only=True)
class PlotDictConfig(BaseConfig):
    """Container for multiple plot configs enabling flexible plot composition.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    plots: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: plots."},
    )
    backend: str = "yellowbrick"

    def __post_init__(self):
        if not self.plots:
            self.plots = {}
        self.backend = normalize_plot_backend(self.backend)

    def __iter__(self):
        return iter(self.plots.items())

    def __len__(self):
        return len(self.plots)

    def merge(self, other: "PlotDictConfig") -> "PlotDictConfig":
        """Merge another PlotDictConfig's plots into this one.

        Later configs override earlier ones with same key.

        Args:
            other: Plot config to merge.

        Returns:
            Self (mutated).
        """
        if isinstance(other, PlotDictConfig):
            self.plots = {**self.plots, **other.plots}
        return self
