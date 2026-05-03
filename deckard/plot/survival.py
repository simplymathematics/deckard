"""Survival-specific seaborn plotting configs.

This module contains the survival plotting classes that used to live in
``deckard.plot.seaborn_plots`` and are now split into a dedicated module.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd

from ..utils import ConfigBase
from .seaborn_plots import SeabornPlotConfig


@dataclass(kw_only=True)
class SurvivalSeabornPlotterConfig(ConfigBase):
    """Factory for seaborn plot configs commonly used in survival model reporting."""

    coefficients_file: Optional[str] = None
    calibration_file: Optional[str] = None
    coefficients_title: str = "Covariate P-values"
    calibration_title: str = "Survival Calibration"

    @staticmethod
    def _resolve_output_path(folder: str, file: str, filetype: str) -> Path:
        file_path = Path(file)
        if file_path.suffix == "":
            file_path = file_path.with_suffix(filetype)
        return Path(folder) / file_path

    def build_coefficients_plot(
        self,
        summary: pd.DataFrame,
    ) -> SeabornPlotConfig:
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

    def build_calibration_plot(
        self,
        calibration: pd.DataFrame,
    ) -> SeabornPlotConfig:
        calibration_df = pd.DataFrame(calibration).copy()
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
        aft: Any,
        title: str,
        file: str,
        xlabel: str,
        ylabel: str,
        replacement_dict: dict,
        dummy_dict: dict,
        folder: str,
        filetype: str = ".pdf",
    ):
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

        if len(selected_columns) == 0:
            ax = aft.plot()
        else:
            ax = aft.plot(columns=selected_columns)

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
        plt.close(ax.figure)
        return ax

    def plot_summary(
        self,
        *,
        aft: Any,
        title: str,
        file: str,
        xlabel: str,
        ylabel: str,
        replacement_dict: dict,
        dummy_dict: dict,
        folder: str,
        filetype: str = ".pdf",
    ):
        file_path = self._resolve_output_path(folder, file, filetype)

        summary = aft.summary.copy().reset_index()
        if "covariate" not in summary.columns and "index" in summary.columns:
            summary = summary.rename(columns={"index": "covariate"})
        if "param" in summary.columns:
            summary = summary[summary["param"] == "lambda_"]
        if "covariate" in summary.columns:
            summary = summary[summary["covariate"] != "Intercept"]

        if replacement_dict:
            summary["covariate"] = summary["covariate"].replace(
                replacement_dict,
            )

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
        ax.set_yscale("log")
        plt.close(ax.figure)
        return ax

    def plot_qq(
        self,
        *,
        aft: Any,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame],
        t0: float,
        title: str,
        file: str,
        xlabel: str,
        ylabel: str,
        calibration_fn: Callable[
            [Any, pd.DataFrame, Optional[pd.DataFrame], float],
            pd.DataFrame,
        ],
        folder: str,
        ax: Optional[Axes] = None,
        filetype: str = ".pdf",
    ):
        calibration_data = calibration_fn(aft, X_train, X_test, t0)
        file_path = self._resolve_output_path(folder, file, filetype)
        cfg = self.build_calibration_plot(calibration_data)
        cfg.title = title
        cfg.xlabel = xlabel
        cfg.ylabel = ylabel
        cfg.kwargs = {"marker": "o"}
        cfg.hue = "dataset" if "dataset" in calibration_data.columns else None
        cfg.plot_file = file_path.as_posix()
        ax = cfg(ax=ax)
        line_min = calibration_data[["predicted", "observed"]].min().min()
        line_max = calibration_data[["predicted", "observed"]].max().max()
        ax.plot([line_min, line_max], [line_min, line_max], "k--", alpha=0.7)
        plt.close(ax.figure)
        return ax

    def plot_partial_effects(
        self,
        *,
        aft: Any,
        covariate_array: list,
        values: list,
        title: str,
        file: str,
        xlabel: str,
        ylabel: str,
        folder: str,
        filetype: str = ".pdf",
    ):
        file_path = self._resolve_output_path(folder, file, filetype)
        ax = aft.plot_partial_effects_on_outcome(covariate_array, values=values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(file_path)
        plt.close(ax.figure)
        return ax


@dataclass(eq=False)
class SurvivalSeabornPlotConfigList(ConfigBase):
    """Container for multiple survival model plots from SurvivalSeabornPlotterConfig."""

    plots_by_model: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    t0s: dict = field(default_factory=dict)
    runtime_data: Any = None

    def add_model_plots(
        self,
        model_type: str,
        model: Any,
        plots: list,
        t0: float = 0.35,
    ) -> None:
        """Add plots for a single model variant."""
        self.plots_by_model[model_type] = plots
        self.models[model_type] = model
        self.t0s[model_type] = t0

    def render_all(self) -> dict:
        """Render all collected plots and return results."""
        results = {
            "by_model": {},
            "all_plots": [],
        }
        for model_type, plots in self.plots_by_model.items():
            results["by_model"][model_type] = plots
            results["all_plots"].extend(plots)
        return results

    def orchestrate_survival_models(
        self,
        model_config: dict,
        data: pd.DataFrame,
        duration_col: str,
        target: str,
        dataset: Optional[str],
        test_size: float = 0.25,
        folder: str = ".",
        dummy_dict: Optional[dict] = None,
    ) -> dict:
        """Orchestrate all survival model experiments: fit, plot, collect results."""
        from ..data import DataConfig
        from ..experiment import SurvivalExperimentConfig
        from ..layers.survival import run_survival_model_experiment

        dummy_dict = dummy_dict or {}

        if not isinstance(test_size, float) or not (0 < test_size < 1):
            raise ValueError("test_size must be a float between 0 and 1")
        if target not in data.columns:
            raise ValueError(f"{target} not in data columns")
        if duration_col not in data.columns:
            raise ValueError(f"{duration_col} not in data columns")

        runtime_data = DataConfig(
            dataset_name="make_regression",
            target=target,
            classifier=False,
            stratify=False,
            train_size=(1 - test_size),
            test_size=test_size,
            random_state=42,
        )
        runtime_data._X = data.drop(columns=[target]).reset_index(drop=True)
        runtime_data._y = data[target].reset_index(drop=True)
        runtime_data.data_load_time = 0.0
        runtime_data._sample()

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
            model, model_plots = run_survival_model_experiment(
                t0=t0,
                mtype=mtype,
                config=cfg,
                X_train=X_train,
                X_test=X_test,
                target=target,
                dummy_dict=dummy_dict,
                duration_col=duration_col,
                folder=folder,
            )
            self.add_model_plots(mtype, model, model_plots, t0)

        survival_config = SurvivalExperimentConfig(
            data=DataConfig(dataset_name=dataset or "toy"),
            duration_col=duration_col,
        )
        summary_table = survival_config.make_survival_model_table(
            self.models,
            dataset,
            X_train,
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
