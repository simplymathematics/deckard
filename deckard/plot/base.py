"""Core plotting primitives and default plotter profiles.

This module provides the _Mixin -> Plugin -> Config pattern for plotting,
enabling flexible composition of plot configurations with experiment state.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from ..utils import BaseConfig, resolve_class

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

    Initialization parameters
    -------------------------
    runtime : Any
        Runtime config object owned by plot config ``__call__``. Mixins should
        treat this as the source of mutable runtime state (experiment state,
        cached results, plot parameters, etc).

    Runtime parameters
    -------------------
    The mixin forwards attribute access to runtime to enable transparent delegation
    of plot configuration to the underlying runtime instance.

    Plugin pattern
    --------------
    Plotter mixins are resolved via PlotTypePlugin and bound to plot configs at
    runtime, enabling flexible backend selection and behavioral extension.
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
class PlotTypePlugin:
    """Generic plotter plugin that binds one mixin to one plotting family/backend.

    Initialization fields
    ---------------------
    mixin_type : Any
        Mixin class (or import path) implementing runtime ``__call__``.
    plot_backend : str
        Backend this plugin matches (e.g., "seaborn", "yellowbrick").
    plot_family : str | None
        Optional plot family constraint (e.g., "feature", "classifier", "regressor").
    excluded_families : tuple[str, ...]
        Families explicitly excluded from this plugin match.
    init_params : dict[str, Any]
        Metadata-only declaration payload for class/type/library docs.

    Plugin hooks
    ------------
    - ``resolve_plotter_mixins`` contributes mixins to runtime plotter context assembly.
    - ``resolve_plotter_handler`` returns callable handler for dispatch.
    - ``__call__`` forwards ``*args``/``**kwargs`` to the configured mixin instance
      bound to the runtime config.
    """

    mixin_type: Any
    plot_backend: str
    plot_family: Union[str, None] = None
    excluded_families: tuple[str, ...] = field(default_factory=tuple)
    init_params: dict[str, Any] = field(default_factory=dict)

    def _resolve_mixin_type(self) -> type:
        if isinstance(self.mixin_type, str):
            resolved = resolve_class(self.mixin_type)
            self.mixin_type = resolved
            return resolved
        return self.mixin_type

    def _matches(
        self,
        *,
        plot_backend: str,
        plot_family: Union[str, None],
    ) -> bool:
        if (plot_backend or "").lower() != (self.plot_backend or "").lower():
            return False
        family = (plot_family or "").lower()
        if self.plot_family is not None and family != self.plot_family.lower():
            return False
        if family in {item.lower() for item in self.excluded_families}:
            return False
        return True

    def resolve_plotter_mixins(
        self,
        runtime: "PlotDictConfig",
        *,
        plot_backend: str,
        plot_family: Union[str, None],
        default_mixins: tuple[type, ...],
    ) -> tuple[type, ...]:
        """Return mixin tuple for matching plotting backend/family.

        Args:
            runtime: Active runtime plot config.
            plot_backend: Requested plotting backend.
            plot_family: Requested plotting family.
            default_mixins: Default mixins for this plotting context.

        Returns:
            Mixin tuple to attach to runtime context.
        """
        _ = (runtime, default_mixins)
        if not self._matches(
            plot_backend=plot_backend,
            plot_family=plot_family,
        ):
            return ()
        mixin = self._resolve_mixin_type()
        return (mixin,)

    def resolve_plotter_handler(
        self,
        runtime: "PlotDictConfig",
        *,
        plot_backend: str,
        plot_family: Union[str, None],
        default_handler: Callable[..., PlotResult] | None,
        default_mixins: tuple[type, ...],
    ) -> Callable[..., PlotResult] | None:
        """Return callable runtime handler for matching backend/family.

        Args:
            runtime: Active runtime plot config.
            plot_backend: Requested plotting backend.
            plot_family: Requested plotting family.
            default_handler: Existing resolved runtime handler.
            default_mixins: Existing resolved mixins.

        Returns:
            Callable runtime plot handler when plugin matches, else None.
        """
        _ = (default_handler, default_mixins)
        if not self._matches(
            plot_backend=plot_backend,
            plot_family=plot_family,
        ):
            return None
        return lambda *args, **kwargs: self(runtime, *args, **kwargs)

    def __call__(
        self,
        runtime: "PlotDictConfig",
        *args: PlotValue,
        **kwargs: PlotValue,
    ) -> PlotResult:
        """Delegate runtime plotter execution to configured mixin handler.

        Args:
            runtime: Runtime config instance currently orchestrating plotting.
            *args: Positional runtime args forwarded to mixin ``__call__``.
            **kwargs: Keyword runtime args forwarded to mixin ``__call__``.

        Returns:
            Rendered plot payload from the configured mixin handler.
        """
        mixin = self._resolve_mixin_type()
        handler = mixin(runtime)
        return handler(*args, **kwargs)


@dataclass(eq=False, kw_only=True)
class PlotDictConfig(BaseConfig):
    """Container for multiple plot configs enabling flexible plot composition.

    Initialization parameters
    -------------------------
    plots : dict[str, Any]
        Named plot configurations (e.g., "feature_pca", "roc_auc").
    plot_backend : str
        Backend selector ("seaborn" or "yellowbrick").

    Runtime parameters
    -------------------
    experiment : Any
        ExperimentConfig instance providing model/data/attack context.

    Plugin pattern
    --------------
    This container participates in PlotTypePlugin-based resolution for
    plot-backend dispatch and mixin composition.
    """

    plots: dict[str, Any] = field(default_factory=dict)
    plot_backend: str = "yellowbrick"

    def __post_init__(self):
        if not self.plots:
            self.plots = {}

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
