"""Survival-specific seaborn plotting configs.

This module contains the survival plotting classes that used to live in
``deckard.plugins.seaborn.plot`` and are now split into a dedicated module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...data import DataConfig
from ...experiment import SurvivalExperimentConfig
from ...frameworks.types import StringifiedClass
from ...utils import BaseConfig
from ..seaborn.plot import SeabornPlotConfig


def _close_axis_figure(ax: Any) -> None:
    fig = getattr(ax, "figure", None)
    if isinstance(fig, Figure):
        plt.close(fig)


PlotScalar = str | int | float | bool | None
PlotValue = PlotScalar | list["PlotValue"] | dict[str, "PlotValue"]


class SurvivalFitterLike(Protocol):
    """Structural protocol for fitted survival estimators used by plot helpers."""

    summary: pd.DataFrame

    def plot(self, *args: Any, **kwargs: Any) -> Axes:
        """Render fitter plot output.

        Args:
            *args: Positional plotting arguments.
            **kwargs: Keyword plotting arguments.

        Returns:
            Matplotlib axis for the rendered fitter plot.
        """
        ...

    def plot_partial_effects_on_outcome(
        self,
        covariates: list[PlotScalar],
        *,
        values: list[PlotScalar],
    ) -> Axes:
        """Render partial-effects curves on fitted survival outcomes.

        Args:
            covariates: Covariates to vary.
            values: Values to evaluate for covariates.

        Returns:
            Matplotlib axis for the rendered partial-effects plot.
        """
        ...


@dataclass(kw_only=True)
class SurvivalSeabornPlotterConfig(BaseConfig):
    """Factory for seaborn plot configs commonly used in survival model reporting."""

    coefficients_file: Optional[str] = None
    calibration_file: Optional[str] = None
    coefficients_title: str = "Covariate P-values"
    calibration_title: str = "Survival Calibration"

    @staticmethod
    def _resolve_output_path(folder: str, file: str, filetype: str) -> Path:
        """Build an output path under a folder with a default suffix.

        Args:
            folder: Destination folder for artifacts.
            file: File name or relative path.
            filetype: Default suffix applied when ``file`` has no suffix.

        Returns:
            Fully resolved output path.
        """
        file_path = Path(file)
        if file_path.suffix == "":
            file_path = file_path.with_suffix(filetype)
        return Path(folder) / file_path

    def build_coefficients_plot(self, summary: pd.DataFrame) -> SeabornPlotConfig:
        """Create a bar-plot config for model coefficient p-values.

        Args:
            summary: Lifelines summary dataframe from a fitted model.

        Returns:
            A seaborn plot config for coefficient p-value bars.
        """
        summary_df = pd.DataFrame(summary).copy()
        if isinstance(summary_df.index, pd.MultiIndex):
            summary_df["covariate"] = summary_df.index.get_level_values(1)
        else:
            summary_df["covariate"] = summary_df.index
        summary_df = summary_df[summary_df["covariate"] != "Intercept"]
        summary_df = summary_df[
            ~summary_df["covariate"].astype(str).str.startswith("dummy_")
        ]
        return SeabornPlotConfig(
            plot_type="bar",
            data=summary_df,
            x="covariate",
            y="p",
            title=self.coefficients_title,
            xlabel="Covariate",
            ylabel="p-value",
            kwargs={"errorbar": None},
            plot_file=self.coefficients_file,
        )

    def build_calibration_plot(self, calibration: pd.DataFrame) -> SeabornPlotConfig:
        """Create a line-plot config for calibration curves.

        Args:
            calibration: Calibration dataframe with predicted and observed columns.

        Returns:
            A seaborn plot config for calibration rendering.

        Raises:
            ValueError: If calibration data is empty.
        """
        calibration_df = pd.DataFrame(calibration).copy()
        if calibration_df.empty:
            raise ValueError("Calibration data is empty")
        return SeabornPlotConfig(
            plot_type="line",
            data=calibration_df,
            x="predicted",
            y="observed",
            title=self.calibration_title,
            xlabel="Predicted Probability",
            ylabel="Observed Probability",
            kwargs={"marker": "o"},
            plot_file=self.calibration_file,
        )

    def plot_aft(
        self,
        *,
        aft: SurvivalFitterLike,
        title: str,
        file: str,
        xlabel: str,
        ylabel: str,
        replacement_dict: Mapping[str, str],
        dummy_dict: Mapping[str, str],
        folder: str,
        filetype: str = ".pdf",
    ) -> Axes:
        """Render and save a fitted-model coefficient plot.

        Args:
            aft: Fitted lifelines model.
            title: Plot title.
            file: Output file name.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            replacement_dict: Label replacement mapping.
            dummy_dict: Dummy-prefix mapping used to filter columns.
            folder: Output folder.
            filetype: Default output suffix.

        Returns:
            The rendered matplotlib axis.

        Raises:
            ValueError: If the fitted summary cannot be rendered.
        """
        file_path = self._resolve_output_path(folder, file, filetype)

        try:
            columns = list(aft.summary.index.get_level_values(1)).copy()
        except Exception:
            columns = list(aft.summary.index).copy()

        dummy_prefixes = tuple(dummy_dict.values())
        selected_columns = []
        for col in columns:
            if str(col).startswith("Intercept"):
                continue
            if len(dummy_prefixes) > 0 and str(col).startswith(dummy_prefixes):
                continue
            selected_columns.append(col)

        ax = (
            aft.plot()
            if len(selected_columns) == 0
            else aft.plot(columns=selected_columns)
        )

        labels = [label.get_text() for label in ax.get_yticklabels()]
        for old, new in replacement_dict.items():
            labels = [label.replace(old, new) for label in labels]
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.get_figure().tight_layout()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ax.get_figure().savefig(file_path)
        _close_axis_figure(ax)
        return ax

    def plot_summary(
        self,
        *,
        aft: SurvivalFitterLike,
        title: str,
        file: str,
        xlabel: str,
        ylabel: str,
        replacement_dict: Mapping[str, str],
        dummy_dict: Mapping[str, str],
        folder: str,
        filetype: str = ".pdf",
    ) -> Axes:
        """Render and save a summary p-value bar plot.

        Args:
            aft: Fitted lifelines model.
            title: Plot title.
            file: Output file name.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            replacement_dict: Label replacement mapping.
            dummy_dict: Optional dummy replacement mapping.
            folder: Output folder.
            filetype: Default output suffix.

        Returns:
            The rendered matplotlib axis.

        Raises:
            ValueError: If summary plot rendering fails.
        """
        file_path = self._resolve_output_path(folder, file, filetype)

        summary = aft.summary.copy().reset_index()
        if "covariate" not in summary.columns and "index" in summary.columns:
            summary = summary.rename(columns={"index": "covariate"})
        if "param" in summary.columns:
            summary = summary[summary["param"] == "lambda_"]
        if "covariate" in summary.columns:
            summary = summary[summary["covariate"] != "Intercept"]
        if replacement_dict:
            summary["covariate"] = summary["covariate"].replace(replacement_dict)
        if dummy_dict:
            summary["covariate"] = summary["covariate"].replace(dummy_dict)

        cfg = SeabornPlotConfig(
            x="covariate",
            y="p",
            kwargs={},
            plot_type="bar",
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            data=summary,
            plot_file=file_path.as_posix(),
        )
        ax = cfg()
        if ax is None:
            raise ValueError("Failed to render summary plot")
        ax.set_yscale("log")
        _close_axis_figure(ax)
        return ax

    def plot_qq(
        self,
        *,
        aft: SurvivalFitterLike,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame],
        t0: float,
        title: str,
        file: str,
        xlabel: str,
        ylabel: str,
        calibration_fn: Callable[
            [SurvivalFitterLike, pd.DataFrame, Optional[pd.DataFrame], float],
            pd.DataFrame,
        ],
        folder: str,
        ax: Optional[Axes] = None,
        filetype: str = ".pdf",
    ) -> Axes:
        """Render and save survival calibration (QQ-style) curves.

        Args:
            aft: Fitted lifelines model.
            X_train: Training dataframe.
            X_test: Optional test dataframe.
            t0: Calibration horizon.
            title: Plot title.
            file: Output file name.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            calibration_fn: Callable that returns a calibration dataframe.
            folder: Output folder.
            ax: Optional existing axis.
            filetype: Default output suffix.

        Returns:
            The rendered matplotlib axis.

        Raises:
            ValueError: If calibration plot rendering fails.
        """
        calibration_data = calibration_fn(aft, X_train, X_test, t0)
        file_path = self._resolve_output_path(folder, file, filetype)
        cfg = self.build_calibration_plot(calibration_data)
        cfg.title = title
        cfg.xlabel = xlabel
        cfg.ylabel = ylabel
        cfg.kwargs = {"marker": "o"}
        cfg.hue = "dataset" if "dataset" in calibration_data.columns else None
        cfg.plot_file = file_path.as_posix()
        rendered_ax = cfg(ax=ax)
        if rendered_ax is None:
            raise ValueError("Failed to render calibration plot")
        line_min = calibration_data[["predicted", "observed"]].min().min()
        line_max = calibration_data[["predicted", "observed"]].max().max()
        rendered_ax.plot([line_min, line_max], [line_min, line_max], "k--", alpha=0.7)
        _close_axis_figure(rendered_ax)
        return rendered_ax

    def plot_partial_effects(
        self,
        *,
        aft: SurvivalFitterLike,
        covariate_array: list[PlotScalar],
        values: list[PlotScalar],
        title: str,
        file: str,
        xlabel: str,
        ylabel: str,
        folder: str,
        filetype: str = ".pdf",
    ) -> Axes:
        """Render and save partial-effects plots for a fitted model.

        Args:
            aft: Fitted lifelines model.
            covariate_array: Covariates passed to lifelines partial-effects API.
            values: Covariate values used to evaluate effects.
            title: Plot title.
            file: Output file name.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            folder: Output folder.
            filetype: Default output suffix.

        Returns:
            The rendered matplotlib axis.

        Raises:
            ValueError: If partial-effects plot rendering fails.
        """
        file_path = self._resolve_output_path(folder, file, filetype)
        ax = aft.plot_partial_effects_on_outcome(covariate_array, values=values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(file_path)
        _close_axis_figure(ax)
        return ax

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[SurvivalFitterLike, list[Axes]]:
        """Render a single model experiment when the plotter config is called.

        Args:
            *args: Positional shorthand runtime arguments.
            **kwargs: Keyword runtime arguments for single-model rendering.

        Returns:
            Fitted survival model and rendered plot axes.
        """
        if args:
            kwargs = {
                "mtype": args[0],
                "config": args[1],
                "X_train": args[2],
                "target": args[3],
                "duration_col": args[4],
                "t0": args[5],
                "X_test": args[6] if len(args) > 6 else None,
                "dummy_dict": args[7] if len(args) > 7 else None,
                "folder": args[8] if len(args) > 8 else ".",
            }
        return self._render_single_model(**kwargs)

    def _render_single_model(
        self,
        mtype: str,
        config: dict[str, PlotValue],
        X_train: pd.DataFrame,
        target: str,
        duration_col: str,
        t0: float,
        X_test: Optional[pd.DataFrame] = None,
        dummy_dict: Optional[dict[str, str]] = None,
        folder: str = ".",
    ) -> tuple[SurvivalFitterLike, list[Axes]]:
        dummy_dict = dummy_dict or {}
        config = dict(config or {})
        plots = []

        plot_dict = dict(config.pop("plot", {}))
        label_dict = dict(config.pop("labels", {}))
        partial_effect_list = list(config.pop("partial_effect", []))
        model_config = dict(config.pop("model", {}))
        model_config.update(config)

        aft = SurvivalExperimentConfig.fit_aft(
            summary_file=plot_dict.get("summary_file", f"{mtype}_summary.csv"),
            folder=folder,
            df=X_train,
            event_col=target,
            duration_col=duration_col,
            mtype=mtype,
            **model_config,
        )

        plots.append(
            self.plot_aft(
                aft=aft,
                title=plot_dict.get("title", mtype.replace("_", " ").title()),
                file=plot_dict.get("plot", f"{mtype}_aft.pdf"),
                xlabel=label_dict.pop("xlabel", "log(theta)"),
                ylabel=label_dict.pop("ylabel", ""),
                replacement_dict=label_dict,
                dummy_dict=dummy_dict,
                folder=folder,
            ),
        )

        plots.append(
            self.plot_qq(
                aft=aft,
                X_train=X_train,
                X_test=X_test,
                t0=t0,
                title=plot_dict.get(
                    "qq_title",
                    f"{mtype.replace('_', ' ').title()} Calibration",
                ),
                file=plot_dict.get("qq_file", f"{mtype}_qq.pdf"),
                xlabel="Predicted Probability",
                ylabel="Observed Probability",
                calibration_fn=lambda model, frame, frame_test, cutoff: pd.concat(
                    [
                        SurvivalExperimentConfig.survival_probability_calibration(
                            model,
                            frame,
                            t0=cutoff,
                            return_curve=True,
                            plot=False,
                        )[3].assign(dataset="train"),
                        *(
                            [
                                SurvivalExperimentConfig.survival_probability_calibration(
                                    model,
                                    frame_test,
                                    t0=cutoff,
                                    return_curve=True,
                                    plot=False,
                                )[
                                    3
                                ].assign(
                                    dataset="test",
                                ),
                            ]
                            if frame_test is not None
                            else []
                        ),
                    ],
                    ignore_index=True,
                ),
                folder=folder,
            ),
        )

        if plot_dict.get("summary_plot") is not None:
            plots.append(
                self.plot_summary(
                    aft=aft,
                    title=plot_dict.get(
                        "summary_title",
                        f"{mtype.replace('_', ' ').title()} P-values",
                    ),
                    file=plot_dict["summary_plot"],
                    xlabel=label_dict.get("summary_xlabel", "Covariate"),
                    ylabel=label_dict.get("summary_ylabel", "p-value"),
                    replacement_dict=label_dict,
                    dummy_dict={},
                    folder=folder,
                ),
            )

        for partial_effect_dict in partial_effect_list:
            effect_config = dict(partial_effect_dict)
            file = effect_config.pop("file", "partial_effects.pdf")
            plots.append(
                self.plot_partial_effects(
                    aft=aft,
                    file=file,
                    folder=folder,
                    **effect_config,
                ),
            )

        return aft, plots

    def run_survival_model_experiment(
        self,
        mtype: str,
        config: dict[str, PlotValue],
        X_train: pd.DataFrame,
        target: str,
        duration_col: str,
        t0: float,
        X_test: Optional[pd.DataFrame] = None,
        dummy_dict: Optional[dict[str, str]] = None,
        folder: str = ".",
    ) -> tuple[SurvivalFitterLike, list[Axes]]:
        """Backward-compatible alias for single-model survival rendering.

        Args:
            mtype: Survival model type token.
            config: Per-model plotting/model configuration mapping.
            X_train: Training dataframe.
            target: Event indicator column.
            duration_col: Duration/time column.
            t0: Calibration horizon.
            X_test: Optional test dataframe.
            dummy_dict: Optional dummy replacement mapping.
            folder: Output folder for generated artifacts.

        Returns:
            Fitted survival model and rendered plot axes.
        """
        return self._render_single_model(
            mtype=mtype,
            config=config,
            X_train=X_train,
            target=target,
            duration_col=duration_col,
            t0=t0,
            X_test=X_test,
            dummy_dict=dummy_dict,
            folder=folder,
        )


@dataclass(eq=False, kw_only=True)
class SurvivalSeabornPlotConfigList(BaseConfig):
    """Container for multiple survival model plots from SurvivalSeabornPlotterConfig."""

    plots_by_model: dict[str, list[Any]] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    t0s: dict[str, float] = field(default_factory=dict)
    runtime_data: Any = None

    def add_model_plots(
        self,
        model_type: StringifiedClass,
        model: Any,
        plots: list[Any],
        t0: float = 0.35,
    ) -> None:
        """Store rendered model artifacts for later table/plot assembly.

        Args:
            model_type: Model key used in configuration.
            model: Fitted model object.
            plots: Rendered plot artifacts for the model.
            t0: Calibration horizon used for this model.
        """
        self.plots_by_model[model_type] = plots
        self.models[model_type] = model
        self.t0s[model_type] = t0

    def render_all(self) -> dict[str, Any]:
        """Flatten stored plots into grouped and aggregate views.

        Returns:
            Mapping containing per-model and merged plot lists.
        """
        results: dict[str, Any] = {"by_model": {}, "all_plots": []}
        for model_type, plots in self.plots_by_model.items():
            results["by_model"][model_type] = plots
            results["all_plots"].extend(plots)
        return results

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Render all configured model variants when the config list is called.

        Args:
            *args: Positional shorthand runtime arguments.
            **kwargs: Keyword runtime arguments for multi-model rendering.

        Returns:
            Rendered plot artifacts grouped by model.
        """
        if args:
            kwargs = {
                "model_config": args[0],
                "data": args[1],
                "survival_config": args[2],
                "dataset": args[3],
                "test_size": args[4] if len(args) > 4 else 0.25,
                "folder": args[5] if len(args) > 5 else ".",
                "dummy_dict": args[6] if len(args) > 6 else None,
            }
        return self._render_all_models(**kwargs)

    def _render_all_models(
        self,
        model_config: Mapping[str, Any],
        data: pd.DataFrame,
        survival_config: "SurvivalExperimentConfig",
        dataset: Optional[str],
        test_size: float = 0.25,
        folder: str = ".",
        dummy_dict: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Render all configured survival models and build a comparison table.

        Args:
            model_config: Mapping of model names to per-model config blocks.
            data: Input dataframe used to build train/test splits.
            survival_config: Parent survival experiment config.
            dataset: Dataset label used for runtime data metadata.
            test_size: Holdout fraction in (0, 1).
            folder: Output folder for generated artifacts.
            dummy_dict: Optional dummy-prefix mapping for plotting.

        Returns:
            Mapping with fitted models, plots, summary table, and runtime data.
        """
        dummy_dict = dict(dummy_dict or {})
        plotter = SurvivalSeabornPlotterConfig()

        target = survival_config.target
        duration_col = survival_config.duration_col

        if data is None:
            raise ValueError("data must be provided for survival plotting")
        if not isinstance(test_size, float) or not (0 < test_size < 1):
            raise ValueError("test_size must be a float between 0 and 1")
        if target not in data.columns:
            raise ValueError(f"{target} not in data columns")
        if duration_col not in data.columns:
            raise ValueError(f"{duration_col} not in data columns")

        runtime_data = DataConfig(
            dataset_name=dataset or "provided_data",
            target=target,
            classifier=False,
            sampler={
                "name": "deckard.data.sample.SplitSampler",
                "train_size": (1 - test_size),
                "test_size": test_size,
                "random_state": 42,
                "stratify": False,
            },
        )
        runtime_data._X = data.drop(columns=[target]).reset_index(drop=True)
        runtime_data._y = data[target].reset_index(drop=True)
        runtime_data.data_load_time = 0.0
        runtime_data.fit()

        if (
            runtime_data.X_train is None
            or runtime_data.X_test is None
            or runtime_data.y_train is None
            or runtime_data.y_test is None
        ):
            raise ValueError(
                "Runtime survival split did not produce train/test partitions",
            )

        X_train = pd.DataFrame(runtime_data.X_train).copy()
        X_test = pd.DataFrame(runtime_data.X_test).copy()
        X_train[target] = runtime_data.y_train.values
        X_test[target] = runtime_data.y_test.values
        X_train = X_train.dropna(axis=0, how="any")
        X_test = X_test.dropna(axis=0, how="any")

        for mtype, sub_config in model_config.items():
            cfg = dict(sub_config or {})
            t0 = cfg.pop("t0", 0.35)
            model, model_plots = plotter(
                mtype=mtype,
                config=cfg,
                X_train=X_train,
                X_test=X_test,
                target=target,
                duration_col=duration_col,
                t0=t0,
                dummy_dict=dummy_dict,
                folder=folder,
            )
            self.add_model_plots(mtype, model, model_plots, t0)

        default_model = next(iter(model_config.keys()), "weibull")
        survival_config = SurvivalExperimentConfig(
            data=DataConfig(dataset_name=dataset or "provided_data"),
            model=default_model,
            target=target,
            duration_col=duration_col,
            event_col=target,
        )
        summary_table = survival_config.make_survival_model_table(
            self.models,
            X_test,
            folder=folder,
            t0s=self.t0s,
        )

        self.runtime_data = runtime_data
        return {
            "models": self.models,
            "plots": self.plots_by_model,
            "table": summary_table,
            "runtime_data": runtime_data,
        }

    def orchestrate_survival_models(
        self,
        model_config: Mapping[str, Any],
        data: pd.DataFrame,
        survival_config: "SurvivalExperimentConfig",
        dataset: Optional[str],
        test_size: float = 0.25,
        folder: str = ".",
        dummy_dict: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Backward-compatible alias for multi-model survival rendering.

        Args:
            model_config: Mapping of model names to per-model config blocks.
            data: Input dataframe used to build train/test splits.
            survival_config: Parent survival experiment config.
            dataset: Dataset label used for runtime data metadata.
            test_size: Holdout fraction in (0, 1).
            folder: Output folder for generated artifacts.
            dummy_dict: Optional dummy-prefix mapping for plotting.

        Returns:
            Mapping with fitted models, plots, summary table, and runtime data.
        """
        return self._render_all_models(
            model_config=model_config,
            data=data,
            survival_config=survival_config,
            dataset=dataset,
            test_size=test_size,
            folder=folder,
            dummy_dict=dummy_dict,
        )
