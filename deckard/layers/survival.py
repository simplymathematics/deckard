import logging
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from lifelines import (
    AalenAdditiveFitter,
    CRCSplineFitter,
    CoxPHFitter,
    GeneralizedGammaRegressionFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    PiecewiseExponentialRegressionFitter,
    WeibullAFTFitter,
)
from lifelines.exceptions import ConvergenceError
from lifelines.fitters import RegressionFitter
from lifelines.utils import CensoringType

from ..attack import AttackConfig
from ..data import DataConfig
from ..experiment import SurvivalExperimentConfig
from .compile_results import parse_studies
from ..model import ModelConfig
from ..plot.seaborn_plots import SurvivalSeabornPlotterConfig
from ..utils import create_parser_from_function, save_data

logger = logging.getLogger(__name__)

__all__ = [
    "survival_main",
    "survival_probability_calibration",
    "fit_aft",
    "plot_aft",
    "survival_parser",
]


AFT_MODEL_TYPES = {
    "weibull": WeibullAFTFitter,
    "log_normal": LogNormalAFTFitter,
    "log_logistic": LogLogisticAFTFitter,
    "cox": CoxPHFitter,
    "aalen": AalenAdditiveFitter,
    "gamma": GeneralizedGammaRegressionFitter,
    "exponential": PiecewiseExponentialRegressionFitter,
}


def _ccl(probabilities: np.ndarray) -> np.ndarray:
    return np.log(-np.log(1 - probabilities))


def survival_probability_calibration(
    model: RegressionFitter,
    df: pd.DataFrame,
    t0: float,
    ax=None,
    color: str = "red",
    return_curve: bool = False,
    plot: bool = True,
) -> tuple[Any, float, float] | tuple[Any, float, float, pd.DataFrame]:
    """Compute survival calibration metrics and optionally render a calibration curve."""
    if ax is None:
        _, ax = plt.subplots()

    duration_col = model.duration_col
    event_col = model.event_col
    calibration_df = df.copy()
    for col in calibration_df.columns:
        calibration_df[col] = pd.to_numeric(calibration_df[col], errors="raise")
    calibration_df = calibration_df.dropna()

    predictions_at_t0 = np.clip(
        1 - model.predict_survival_function(calibration_df, times=[t0]).T.squeeze(),
        1e-10,
        1 - 1e-10,
    )

    t0_tag = str(t0).replace(".", "_")
    predictor_col = f"ccl_at_{t0_tag}"
    prediction_df = pd.DataFrame(
        {
            predictor_col: _ccl(predictions_at_t0),
            duration_col: calibration_df[duration_col],
            event_col: calibration_df[event_col],
        },
    )

    regressors = {
        "beta_": [predictor_col],
        "gamma0_": "1",
        "gamma1_": "1",
        "gamma2_": "1",
    }

    crc = CRCSplineFitter(n_baseline_knots=3, penalizer=0.000001)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        try:
            if CensoringType.is_right_censoring(model):
                crc.fit_right_censoring(
                    prediction_df,
                    duration_col,
                    event_col,
                    regressors=regressors,
                )
            elif CensoringType.is_left_censoring(model):
                crc.fit_left_censoring(
                    prediction_df,
                    duration_col,
                    event_col,
                    regressors=regressors,
                )
            elif CensoringType.is_interval_censoring(model):
                crc.fit_interval_censoring(
                    prediction_df,
                    duration_col,
                    event_col,
                    regressors=regressors,
                )
            else:
                crc.fit(
                    prediction_df,
                    duration_col,
                    event_col,
                    regressors=regressors,
                )
        except Exception as error:
            logger.error("Could not fit CRC model for calibration: %s", error)
            if return_curve:
                curve = pd.DataFrame(
                    {"predicted": predictions_at_t0, "observed": np.nan},
                ).sort_values("predicted")
                return ax, np.nan, np.nan, curve
            return ax, np.nan, np.nan

    x = np.linspace(
        np.clip(predictions_at_t0.min() - 0.01, 0, 1),
        np.clip(predictions_at_t0.max() + 0.01, 0, 1),
        100,
    )
    y = (
        1
        - crc.predict_survival_function(
            pd.DataFrame({predictor_col: _ccl(x)}),
            times=[t0],
        ).T.squeeze()
    )
    curve_df = pd.DataFrame({"predicted": x, "observed": y})

    if plot:
        ax.plot(x, y, label="Calibration Curve", color=color)
        ax.plot(x, x, c="k", ls="--")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Probability")
        ax.legend()

    try:
        deltas = (
            (1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze()
            - predictions_at_t0
        ).abs()
        ici = deltas.mean()
        e50 = np.percentile(deltas, 50)
    except Exception as error:
        logger.error("Could not compute calibration deltas: %s", error)
        ici = np.nan
        e50 = np.nan

    if return_curve:
        return ax, ici, e50, curve_df
    return ax, ici, e50


def _initialize_aft_fitter(mtype: str, kwargs: dict) -> RegressionFitter:
    if mtype not in AFT_MODEL_TYPES:
        raise ValueError(
            f"Model type {mtype} not recognized. Supported: {list(AFT_MODEL_TYPES.keys())}",
        )

    params = dict(kwargs)
    if mtype in [
        "weibull",
        "log_normal",
        "log_logistic",
        "cox",
        "gamma",
        "exponential",
    ]:
        params.setdefault("penalizer", 0.1)
    if mtype == "aalen":
        params.setdefault("alpha", 0.1)

    fitter_cls = AFT_MODEL_TYPES[mtype]
    return fitter_cls(**params)


def fit_aft(
    df: pd.DataFrame,
    event_col: str,
    duration_col: str,
    mtype: str,
    summary_file: Optional[str] = None,
    folder: Optional[str] = None,
    **kwargs,
) -> RegressionFitter:
    """Fit a survival model and optionally persist its summary."""
    if duration_col not in df.columns:
        raise ValueError(f"Column {duration_col} not found in data")
    if event_col is not None and event_col not in df.columns:
        raise ValueError(f"Column {event_col} not found in data")

    aft = _initialize_aft_fitter(mtype=mtype, kwargs=kwargs)
    fit_kwargs = {"duration_col": duration_col, "event_col": event_col}
    if mtype != "aalen":
        start = df[duration_col].min()
        end = df[duration_col].max()
        start = start - 0.01 * (end - start)
        fit_kwargs["timeline"] = np.linspace(start, end, 1000)

    try:
        aft.fit(df, **fit_kwargs)
    except (ConvergenceError, AttributeError) as error:
        if "delta contains nan value(s)" in str(error):
            fit_kwargs["fit_options"] = {
                "step_size": 0.1,
                "max_steps": 1000,
                "precision": 1e-3,
            }
        else:
            aft._scipy_fit_method = "SLSQP"
        aft.fit(df, **fit_kwargs)

    if summary_file is not None:
        summary = pd.DataFrame(aft.summary).copy()
        summary_path = Path(summary_file)
        if folder is not None:
            summary_path = Path(folder) / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        save_data(summary, filepath=summary_path.as_posix())

    return aft


def plot_aft(
    aft: RegressionFitter,
    title: str,
    file: str,
    xlabel: str,
    ylabel: str,
    replacement_dict: Optional[dict[str, str]] = None,
    dummy_dict: Optional[dict[str, str]] = None,
    folder: Optional[str] = None,
    filetype: str = ".pdf",
) -> Any:
    """Render and save a coefficient plot for a fitted AFT model."""
    replacement_dict = replacement_dict or {}
    dummy_dict = dummy_dict or {}
    file_path = Path(file)
    if file_path.suffix == "":
        file_path = file_path.with_suffix(filetype)
    if folder is not None:
        file_path = Path(folder) / file_path

    plt.gcf().clear()
    try:
        columns = list(aft.summary.index.get_level_values(1)).copy()
    except IndexError:
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
    plt.gcf().clear()
    return ax


def plot_summary(
    aft: RegressionFitter,
    title: str,
    file: str,
    xlabel: str,
    ylabel: str,
    replacement_dict: Optional[dict[str, str]] = None,
    folder: Optional[str] = None,
    filetype: str = ".pdf",
) -> Any:
    """Render and save a summary p-value plot for a fitted survival model."""
    replacement_dict = replacement_dict or {}
    file_path = Path(file)
    if file_path.suffix == "":
        file_path = file_path.with_suffix(filetype)
    if folder is not None:
        file_path = Path(folder) / file_path

    plotter = SurvivalSeabornPlotterConfig(
        coefficients_file=file_path.as_posix(),
        coefficients_title=title,
    )
    cfg = plotter.build_coefficients_plot(pd.DataFrame(aft.summary).copy())
    ax = cfg()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    labels = [label.get_text() for label in ax.get_xticklabels()]
    for old, new in replacement_dict.items():
        labels = [label.replace(old, new) for label in labels]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yscale("log")
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(file_path)
    plt.gcf().clear()
    return ax


def plot_qq(
    X_train: pd.DataFrame,
    aft: RegressionFitter,
    title: str,
    file: str,
    X_test: Optional[pd.DataFrame] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    folder: Optional[str] = None,
    ax=None,
    filetype: str = ".pdf",
    t0: float = 0.35,
) -> Any:
    """Render and save calibration (QQ-style) curves for train/test datasets."""
    file_path = Path(file)
    if file_path.suffix == "":
        file_path = file_path.with_suffix(filetype)
    if folder is not None:
        file_path = Path(folder) / file_path

    if ax is None:
        _, ax = plt.subplots()

    _, _, _, curve_train = survival_probability_calibration(
        aft,
        X_train,
        t0=t0,
        return_curve=True,
        plot=False,
    )
    curve_train["dataset"] = "train"
    calibration_frames = [curve_train]

    if X_test is not None:
        _, _, _, curve_test = survival_probability_calibration(
            aft,
            X_test,
            t0=t0,
            return_curve=True,
            plot=False,
        )
        curve_test["dataset"] = "test"
        calibration_frames.append(curve_test)

    calibration_data = pd.concat(calibration_frames, ignore_index=True)
    plotter = SurvivalSeabornPlotterConfig(
        calibration_file=file_path.as_posix(),
        calibration_title=title,
    )
    cfg = plotter.build_calibration_plot(calibration_data)
    cfg.hue = "dataset" if "dataset" in calibration_data.columns else None
    cfg.kwargs = {"marker": "o"}
    ax = cfg(ax=ax)
    ax.plot([0, 1], [0, 1], c="k", ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(file_path)
    plt.gcf().clear()
    return ax


def plot_partial_effects(
    aft: RegressionFitter,
    covariate_array: Any,
    values_array: Any,
    title: Optional[str] = None,
    file: str = "partial_effects.pdf",
    xlabel: str = "Covariate",
    ylabel: str = "Failure rate",
    legend_kwargs: Optional[dict[str, Any]] = None,
    replacement_dict: Optional[dict[str, str]] = None,
    cmap: str = "coolwarm",
    folder: str = ".",
    filetype: str = ".pdf",
    **kwargs,
) -> Any:
    """Render and save partial-effects plots for selected covariates."""
    legend_kwargs = legend_kwargs or {"loc": "upper left"}
    replacement_dict = replacement_dict or {}
    file_path = Path(folder, file).with_suffix(filetype)

    plt.gcf().clear()
    ax = aft.plot_partial_effects_on_outcome(
        covariate_array,
        values_array,
        cmap=cmap,
        **kwargs,
    )
    labels = [label.get_text() for label in ax.get_yticklabels()]
    for old, new in replacement_dict.items():
        labels = [label.replace(old, new) for label in labels]
    ax.set_yticklabels(labels)
    ax.legend(**legend_kwargs)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.get_figure().tight_layout()
    ax.get_figure().savefig(file_path)
    plt.gcf().clear()
    return ax


def score_model(
    aft: RegressionFitter,
    train: pd.DataFrame,
    test: pd.DataFrame,
    t0: float = 0.35,
    method: str = "concordance_index",
) -> dict[str, float]:
    """Score a fitted survival model on train/test frames with calibration metrics."""
    train_score = aft.score(train, scoring_method=method)
    test_score = aft.score(test, scoring_method=method)
    _, train_ici, train_e50 = survival_probability_calibration(aft, train, t0=t0)
    _, test_ici, test_e50 = survival_probability_calibration(aft, test, t0=t0)
    return {
        "train_score": train_score,
        "test_score": test_score,
        "train_ici": train_ici,
        "test_ici": test_ici,
        "train_e50": train_e50,
        "test_e50": test_e50,
    }


def make_survival_model_table(
    aft_dict: dict[str, RegressionFitter],
    dataset: Optional[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    t0s: dict[str, float],
    folder: str = ".",
    span_columns: bool = True,
) -> pd.DataFrame:
    """Build and persist a summary comparison table across survival models."""
    folder_path = Path(folder)
    data = pd.DataFrame(index=[k.replace("_", " ").title() for k in aft_dict.keys()])
    data.index.name = "Model"

    def _safe_attr(model, attr, fallback=np.nan):
        try:
            return getattr(model, attr)
        except Exception:
            return fallback

    def _safe_score(model, frame):
        try:
            return model.score(frame, scoring_method="concordance_index")
        except Exception:
            return np.nan

    data["AIC"] = [_safe_attr(model, "AIC_") for model in aft_dict.values()]
    data["BIC"] = [_safe_attr(model, "BIC_") for model in aft_dict.values()]
    data["Concordance"] = [_safe_score(model, X_train) for model in aft_dict.values()]
    data["Test Concordance"] = [
        _safe_score(model, X_test) for model in aft_dict.values()
    ]

    train_icis, train_e50s, test_icis, test_e50s = [], [], [], []
    for mtype, model in aft_dict.items():
        t0 = t0s[mtype]
        _, train_ici, train_e50 = survival_probability_calibration(
            model,
            X_train,
            t0=t0,
        )
        _, test_ici, test_e50 = survival_probability_calibration(model, X_test, t0=t0)
        train_icis.append(train_ici)
        train_e50s.append(train_e50)
        test_icis.append(test_ici)
        test_e50s.append(test_e50)

    data["ICI"] = train_icis
    data["Test ICI"] = test_icis
    data["E50"] = train_e50s
    data["Test E50"] = test_e50s

    data = data.round(4)
    folder_path.mkdir(parents=True, exist_ok=True)
    data.to_csv(folder_path / "aft_comparison.csv")
    latex = data.fillna("--")
    pretty_dataset = (
        dataset.upper() if dataset and dataset.lower() != "combined" else dataset
    )
    latex.to_latex(
        folder_path / "aft_comparison.tex",
        float_format="%.3g",
        label=(f"tab:{dataset.lower()}" if dataset is not None else "tab:aft_models"),
        caption=(
            f"Comparison of AFT Models on the {pretty_dataset} dataset."
            if pretty_dataset is not None
            else None
        ),
        index=True,
        header=True,
    )

    if span_columns:
        tex_file = folder_path / "aft_comparison.tex"
        tex_data = tex_file.read_text()
        tex_data = tex_data.replace("\\begin{table}", "\\begin{table*}\n\\centering")
        tex_data = tex_data.replace("\\end{table}", "\\end{table*}")
        tex_file.write_text(tex_data)

    return data


def clean_data_for_aft(
    data: pd.DataFrame,
    covariate_list: list[str],
    target: str = "adv_failure_rate",
    dummy_dict: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Clean and encode tabular data for AFT-style survival fitting."""
    dummy_dict = dummy_dict or {}
    if target not in data.columns:
        raise ValueError(f"Target {target} not in dataframe")

    selected_columns = list(set(list(covariate_list) + [target]))
    selected_columns = [c for c in selected_columns if c in data.columns]
    subset = data[selected_columns].copy()
    for col in subset.columns:
        subset = subset[subset[col] != -1e10]
        subset = subset[subset[col] != 1e10]

    if len(dummy_dict) > 0:
        available_dummy_cols = [c for c in dummy_dict.keys() if c in subset.columns]
        dummies = pd.get_dummies(
            subset[available_dummy_cols],
            prefix={k: dummy_dict[k] for k in available_dummy_cols},
            prefix_sep=" ",
            columns=available_dummy_cols,
        )
        subset = subset.drop(columns=available_dummy_cols)
        cleaned = pd.concat([subset, dummies], axis=1)
    else:
        cleaned = subset.copy()
        object_cols = [col for col in cleaned.columns if cleaned[col].dtype == "object"]
        if len(object_cols) > 0:
            dummies = pd.get_dummies(cleaned[object_cols], prefix="", prefix_sep="")
            cleaned = cleaned.drop(columns=object_cols)
            cleaned = pd.concat([cleaned, dummies], axis=1)
        cleaned = cleaned.astype(float)

    cleaned = cleaned.dropna(axis=0, how="any")
    if target not in cleaned.columns:
        raise ValueError(f"Target {target} not in cleaned dataframe")
    return cleaned


def _build_runtime_survival_data_config(
    data: pd.DataFrame,
    target: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> DataConfig:
    """Create a DataConfig-backed train/test split from an in-memory survival dataframe."""
    if not isinstance(test_size, float) or not (0 < test_size < 1):
        raise ValueError("test_size must be a float between 0 and 1")
    if target not in data.columns:
        raise ValueError(f"Target {target} not in dataframe")

    runtime_data = DataConfig(
        dataset_name="make_regression",
        target=target,
        classifier=False,
        stratify=False,
        train_size=(1 - test_size),
        test_size=test_size,
        random_state=random_state,
    )
    runtime_data._X = data.drop(columns=[target]).reset_index(drop=True)
    runtime_data._y = data[target].reset_index(drop=True)
    runtime_data.data_load_time = 0.0
    runtime_data._sample()
    return runtime_data


def run_survival_model_experiment(
    mtype: str,
    config: dict[str, Any],
    X_train: pd.DataFrame,
    target: str,
    duration_col: str,
    t0: float,
    X_test: Optional[pd.DataFrame] = None,
    dummy_dict: Optional[dict[str, str]] = None,
    folder: str = ".",
) -> tuple[RegressionFitter, list[Any]]:
    """Fit one survival model variant and render its configured plots."""
    dummy_dict = dummy_dict or {}
    config = dict(config or {})
    plots = []

    plot_dict = dict(config.pop("plot", {}))
    label_dict = dict(config.pop("labels", {}))
    partial_effect_list = list(config.pop("partial_effect", []))
    model_config = dict(config.pop("model", {}))
    model_config.update(config)

    aft = fit_aft(
        summary_file=plot_dict.get("summary_file", f"{mtype}_summary.csv"),
        folder=folder,
        df=X_train,
        event_col=target,
        duration_col=duration_col,
        mtype=mtype,
        **model_config,
    )

    aft_plot = plot_aft(
        aft=aft,
        title=plot_dict.get("title", mtype.replace("_", " ").title()),
        file=plot_dict.get("plot", f"{mtype}_aft.pdf"),
        xlabel=label_dict.pop("xlabel", "log(theta)"),
        ylabel=label_dict.pop("ylabel", ""),
        replacement_dict=label_dict,
        dummy_dict=dummy_dict,
        folder=folder,
    )
    plots.append(aft_plot)

    qq_plot = plot_qq(
        X_train=X_train,
        X_test=X_test,
        aft=aft,
        title=plot_dict.get(
            "qq_title",
            f"{mtype.replace('_', ' ').title()} Calibration",
        ),
        t0=t0,
        file=plot_dict.get("qq_file", f"{mtype}_qq.pdf"),
        xlabel="Predicted Probability",
        ylabel="Observed Probability",
        folder=folder,
    )
    plots.append(qq_plot)

    if plot_dict.get("summary_plot") is not None:
        summary_plot = plot_summary(
            aft=aft,
            title=plot_dict.get(
                "summary_title",
                f"{mtype.replace('_', ' ').title()} P-values",
            ),
            file=plot_dict["summary_plot"],
            xlabel=label_dict.get("summary_xlabel", "Covariate"),
            ylabel=label_dict.get("summary_ylabel", "p-value"),
            replacement_dict=label_dict,
            folder=folder,
        )
        plots.append(summary_plot)

    for partial_effect_dict in partial_effect_list:
        effect_config = dict(partial_effect_dict)
        file = effect_config.pop("file", "partial_effects.pdf")
        partial_effect_plot = plot_partial_effects(
            aft=aft,
            file=file,
            folder=folder,
            **effect_config,
        )
        plots.append(partial_effect_plot)

    return aft, plots


def render_all_survival_model_plots(
    config: dict[str, Any],
    duration_col: str,
    target: str,
    data: pd.DataFrame,
    dataset: Optional[str],
    test_size: float = 0.25,
    folder: str = ".",
    dummy_dict: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Run all configured survival model variants and collect outputs."""
    dummy_dict = dummy_dict or {}
    if target not in data.columns:
        raise ValueError(f"{target} not in data columns")
    if duration_col not in data.columns:
        raise ValueError(f"{duration_col} not in data columns")

    runtime_data = _build_runtime_survival_data_config(
        data=data,
        target=target,
        test_size=test_size,
        random_state=42,
    )
    if (
        runtime_data.X_train is None
        or runtime_data.X_test is None
        or runtime_data.y_train is None
        or runtime_data.y_test is None
    ):
        raise ValueError("Runtime survival split did not produce train/test partitions")

    X_train = pd.DataFrame(runtime_data.X_train).copy()
    X_test = pd.DataFrame(runtime_data.X_test).copy()
    X_train[target] = runtime_data.y_train.values
    X_test[target] = runtime_data.y_test.values
    X_train = X_train.dropna(axis=0, how="any")
    X_test = X_test.dropna(axis=0, how="any")

    models = {}
    plots = {}
    t0s = {}
    for mtype, sub_config in config.items():
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
        models[mtype] = model
        plots[mtype] = model_plots
        t0s[mtype] = t0

    summary_table = make_survival_model_table(
        models,
        dataset,
        X_train,
        X_test,
        folder=folder,
        t0s=t0s,
    )
    return {
        "models": models,
        "plots": plots,
        "table": summary_table,
        "runtime_data": runtime_data,
    }


def _resolve_survival_model_name(model: Union[str, dict, ModelConfig]) -> str:
    if isinstance(model, DictConfig):
        model = OmegaConf.to_container(model, resolve=True)
    if isinstance(model, str):
        return model.lower()
    if isinstance(model, ModelConfig):
        return (model.model_type).lower()
    if isinstance(model, dict):
        return str(
            model.get("survival_model") or model.get("alias") or model.get("model"),
        ).lower()
    raise TypeError(f"Unsupported model specification: {type(model)}")


def _normalize_survival_model_name(model_name: str) -> str:
    normalized = str(model_name).split(".")[-1].lower()
    alias_map = {
        "weibullaftfitter": "weibull",
        "lognormalaftfitter": "log_normal",
        "loglogisticaftfitter": "log_logistic",
        "coxphfitter": "cox",
        "aalenadditivefitter": "aalen",
        "generalizedgammaregressionfitter": "gamma",
        "piecewiseexponentialregressionfitter": "exponential",
    }
    return alias_map.get(normalized, normalized)


def _resolve_survival_runtime_models(model, model_config, attack):
    """Split survival-fitter vs attacked-model semantics.

    No attack: `model` means the survival fitter and no auxiliary attack model is required.
    Attack present: `model`/`model_config` may describe the attacked model, and
    a distinct `survival_model` key becomes meaningful.
    """
    if isinstance(model, DictConfig):
        model = OmegaConf.to_container(model, resolve=True)
    if isinstance(model_config, DictConfig):
        model_config = OmegaConf.to_container(model_config, resolve=True)

    if attack is None:
        if isinstance(model, ModelConfig):
            survival_model = _normalize_survival_model_name(model.model_type)
        elif isinstance(model, dict):
            survival_model = _normalize_survival_model_name(
                model.get("model_type") or model.get("alias") or "weibull",
            )
        else:
            survival_model = _normalize_survival_model_name(model)
        return survival_model, None

    if model_config is not None:
        attack_model_spec = model_config
        if isinstance(model, dict):
            survival_model = _normalize_survival_model_name(
                model.get("survival_model")
                or model.get("alias")
                or model.get("model")
                or model.get("model_type")
                or "weibull",
            )
        else:
            survival_model = _normalize_survival_model_name(
                _resolve_survival_model_name(model),
            )
    else:
        attack_model_spec = model
        if isinstance(model, ModelConfig):
            survival_model = _normalize_survival_model_name(
                getattr(model, "survival_model", None)
                or model.alias
                or model.model_type,
            )
        elif isinstance(model, dict):
            survival_model = _normalize_survival_model_name(
                model.get("survival_model") or model.get("alias") or None
            )
            if survival_model is None:
                raise ValueError("Survival model must be")
        else:
            survival_model = _normalize_survival_model_name(model)

    if isinstance(attack_model_spec, ModelConfig):
        attack_model_spec.classifier = False
        return survival_model, attack_model_spec

    if isinstance(attack_model_spec, dict):
        aux_cfg = dict(attack_model_spec)
        aux_cfg.pop("survival_model", None)
        aux_cfg.setdefault("model_type", "sklearn.linear_model.LinearRegression")
        aux_cfg["classifier"] = False
        return survival_model, ModelConfig(**aux_cfg)

    if attack_model_spec is None:
        raise ValueError("An attack-aware survival run requires a model config")

    return survival_model, ModelConfig(
        model_type="sklearn.linear_model.LinearRegression",
        classifier=False,
        model_params={},
        alias="survival-aux",
    )


def _resolve_data_name(data: Union[str, dict, DataConfig]) -> str:
    if isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=True)
    if isinstance(data, str):
        return data
    if isinstance(data, DataConfig):
        return data.dataset_name
    if isinstance(data, dict):
        return str(data.get("dataset_name", data.get("alias", "lung")))
    raise TypeError(f"Unsupported data specification: {type(data)}")


def _get_attack_label_column(data: pd.DataFrame) -> Optional[str]:
    for candidate in ["attack name", "attack_name", "attack", "attack_alias"]:
        if candidate in data.columns:
            return candidate
    return None


def _infer_attack_kind_from_label(label: Optional[str]) -> Optional[str]:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    value = str(label).strip().lower()
    if value == "":
        return None
    elif any(token in value for token in ["membership", "member"]):
        return "membership"
    elif any(token in value for token in ["attribute", "attr"]):
        return "attribute"
    else:
        return "evasion"


def _candidate_attack_metrics_for_kind(attack_kind: Optional[str]) -> list[str]:
    if attack_kind == "evasion":
        return ["evasion_success", "evasion_accuracy"]
    if attack_kind == "membership":
        return ["membership_inference_accuracy"]
    if attack_kind == "attribute":
        return ["sex_inference_accuracy", "attribute_inference_accuracy"]
    return [
        "evasion_success",
        "evasion_accuracy",
        "membership_inference_accuracy",
        "sex_inference_accuracy",
        "attribute_inference_accuracy",
    ]


def _attack_kind_from_config(attack_config: Optional[AttackConfig]) -> Optional[str]:
    if attack_config is None:
        return None

    attack_path = getattr(attack_config, "attack_type", "") or ""
    parts = attack_path.split("attacks.")[-1].split(".")
    attack_type = parts[0] if len(parts) > 0 else ""
    attack_subtype = parts[1] if len(parts) > 1 else ""
    subtype = attack_subtype.lower()

    if attack_type == "evasion":
        return "evasion"
    if attack_type == "inference" and "membership" in subtype:
        return "membership"
    if attack_type == "inference" and "attribute" in subtype:
        return "attribute"
    return None


def _load_optuna_survival_frame(
    optuna_db: str,
    schema: Optional[Union[str, dict]] = None,
    query: Optional[str] = None,
) -> pd.DataFrame:
    frame = parse_studies(optuna_db=optuna_db, schema=schema or {})
    if query is not None:
        frame = frame.query(query)
    if frame.empty:
        raise ValueError(
            f"No attack results found in {optuna_db} after applying filters",
        )
    return frame


def _resolve_attack_size(
    output: pd.DataFrame,
    row_index: Optional[Any] = None,
    attack_config: Optional[AttackConfig] = None,
) -> float:
    if row_index is not None and "attack_size" in output.columns:
        attack_size = output.at[row_index, "attack_size"]
        if not pd.isna(attack_size):
            return float(attack_size)
    if "attack_size" in output.columns and output["attack_size"].notna().all():
        unique_sizes = output["attack_size"].dropna().unique()
        if len(unique_sizes) == 1:
            return float(unique_sizes[0])
    if attack_config is not None:
        return float(attack_config.attack_size)
    return 1.0


def _failure_count_from_metric(
    value: float,
    metric: str,
    attack_size: float,
) -> float:
    failure_rate = value if metric.endswith("_success") else 1 - value
    return attack_size * failure_rate


def calculate_failures_under_attack(
    data: pd.DataFrame,
    attack_config: Optional[AttackConfig] = None,
    benign_metric: str = "accuracy",
) -> pd.DataFrame:
    """Optionally derive ben/adv failure counts from attack-specific accuracy metrics."""
    output = data.copy()
    if benign_metric in output.columns and "ben_failures" not in output.columns:
        if "attack_size" in output.columns:
            attack_sizes = output["attack_size"].fillna(
                _resolve_attack_size(output, attack_config=attack_config),
            )
        else:
            attack_sizes = pd.Series(
                _resolve_attack_size(output, attack_config=attack_config),
                index=output.index,
                dtype=float,
            )
        output["ben_failures"] = attack_sizes * (1 - output[benign_metric])

    attack_label_col = _get_attack_label_column(output)
    attack_kind = _attack_kind_from_config(attack_config)

    if attack_label_col is not None:
        adv_failures = pd.Series(np.nan, index=output.index, dtype=float)
        for row_index, attack_label in output[attack_label_col].items():
            row_kind = _infer_attack_kind_from_label(attack_label) or attack_kind
            for metric in _candidate_attack_metrics_for_kind(row_kind):
                if metric not in output.columns or pd.isna(
                    output.at[row_index, metric],
                ):
                    continue
                value = output.at[row_index, metric]
                adv_failures.at[row_index] = _failure_count_from_metric(
                    value=value,
                    metric=metric,
                    attack_size=_resolve_attack_size(
                        output,
                        row_index=row_index,
                        attack_config=attack_config,
                    ),
                )
                break
        if adv_failures.notna().any():
            output["adv_failures"] = adv_failures
            return output

    for metric in _candidate_attack_metrics_for_kind(attack_kind):
        if metric in output.columns:
            if "attack_size" in output.columns:
                attack_sizes = output["attack_size"].fillna(
                    _resolve_attack_size(output, attack_config=attack_config),
                )
            else:
                attack_sizes = pd.Series(
                    _resolve_attack_size(output, attack_config=attack_config),
                    index=output.index,
                    dtype=float,
                )
            output["adv_failures"] = attack_sizes * (
                output[metric]
                if metric.endswith("_success")
                else 1 - output[metric]
            )
            break
    return output


def _build_data_config(
    data_name: str,
    target: str,
    test_size: float,
    random_state: int,
) -> DataConfig:
    lifelines_like = {
        "lung",
        "leukemia",
        "diabetes",
        "lifelines_diabetes",
        "lifelines.lung",
        "lifelines.leukemia",
        "lifelines.diabetes",
    }
    # Keep existing `data=diabetes` command intuitive for survival workflows.
    if data_name == "diabetes":
        data_name = "lifelines_diabetes"
    target_for_load = None if data_name in lifelines_like else target
    data_cfg = DataConfig(
        dataset_name=data_name,
        target=target_for_load,
        classifier=False,
        stratify=False,
        test_size=test_size,
        random_state=random_state,
    )
    data_cfg._load_data()
    return data_cfg


def _evaluate_aux_model(model_config: ModelConfig, data_config: DataConfig):
    return model_config(data_config)


def survival_main(
    data: str = "lung",
    model: str = "weibull",
    plots_folder: str = "plots/survival",
    config_file: Optional[str] = None,
    target: str = "E",
    duration_col: str = "T",
    dataset: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
    survival_model: Optional[str] = None,
    attack: Optional[Union[dict[str, Any], AttackConfig]] = None,
    calculate_attack_failures: bool = False,
    data_file: Optional[str] = None,
    attack_optuna_db: Optional[str] = None,
    attack_schema: Optional[Union[str, dict[str, Any]]] = None,
    attack_query: Optional[str] = None,
) -> dict[str, Any]:
    """Run survival model experiments from config and persist plots and summary tables.

    Supports both modern `data`/`model` arguments and legacy `data_file` usage.
    """
    logging.basicConfig(level=logging.INFO)
    matplotlib.rc("font", **{"family": "Times New Roman", "size": 22})

    output_folder = Path(plots_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    config = {}
    if config_file is not None:
        with Path(config_file).open("r") as handle:
            config = yaml.safe_load(handle) or {}

    test_size = float(config.pop("test_size", 0.25))
    random_state = int(config.pop("random_state", 42))
    fillna = dict(config.pop("fillna", {}))
    dummies = dict(config.pop("dummies", {}))
    covariates = list(config.pop("covariates", [duration_col, target]))
    if model_config is None:
        model_config = config.pop("model_config", None)
    if attack is None:
        attack = config.pop("attack", None)
    if attack_optuna_db is None:
        attack_optuna_db = config.pop("attack_optuna_db", None)
    if attack_schema is None:
        attack_schema = config.pop("attack_schema", None)
    if attack_query is None:
        attack_query = config.pop("attack_query", None)

    if data_file is not None:
        data = data_file

    data_spec = data
    data_name = _resolve_data_name(data_spec)
    attack_cfg = AttackConfig(**attack) if isinstance(attack, dict) else attack
    resolved_survival_model, aux_model = _resolve_survival_runtime_models(
        model=model,
        model_config=model_config,
        attack=attack_cfg,
    )
    if survival_model is not None:
        resolved_survival_model = _normalize_survival_model_name(survival_model)

    experiment = None
    if attack_optuna_db is not None:
        loaded_data = _load_optuna_survival_frame(
            optuna_db=attack_optuna_db,
            schema=attack_schema,
            query=attack_query,
        )
    else:
        experiment = SurvivalExperimentConfig(
            data=_build_data_config(
                data_name=data_name,
                target=target,
                test_size=test_size,
                random_state=random_state,
            ),
            model=aux_model,
            attack=attack_cfg,
            survival_model=resolved_survival_model,
            duration_col=duration_col,
            event_col=target,
            classifier=False,
            library="sklearn",
        )

        data_cfg = experiment.data
        loaded_data = data_cfg.X.copy()
        if data_cfg.y is not None and target not in loaded_data.columns:
            loaded_data[target] = data_cfg.y.values

    # Normalize common lifelines column names into the requested target/duration pair.
    if duration_col not in loaded_data.columns:
        for candidate in ["T", "time", "t", "duration", "right"]:
            if candidate in loaded_data.columns:
                duration_col = candidate
                break
    if target not in loaded_data.columns:
        for candidate in ["E", "status", "event"]:
            if candidate in loaded_data.columns:
                target = candidate
                break
    if target not in loaded_data.columns and "right" in loaded_data.columns:
        loaded_data[target] = np.isfinite(loaded_data["right"]).astype(int)
        duration_col = "right"
        if "left" in loaded_data.columns:
            loaded_data[duration_col] = loaded_data["right"].where(
                np.isfinite(loaded_data["right"]),
                loaded_data["left"],
            )

    loaded_data.columns = loaded_data.columns.str.strip()
    for col, value in fillna.items():
        if col not in loaded_data.columns:
            raise ValueError(f"{col} not found in input data")
        loaded_data[col] = loaded_data[col].fillna(value)

    if calculate_attack_failures or target in {"ben_failures", "adv_failures"}:
        loaded_data = calculate_failures_under_attack(
            loaded_data,
            attack_config=attack_cfg,
            benign_metric=config.get("failure_metric", "accuracy"),
        )

    if target not in covariates:
        covariates.append(target)
    if duration_col not in covariates:
        covariates.append(duration_col)

    cleaned = clean_data_for_aft(
        loaded_data,
        covariates,
        target=target,
        dummy_dict=dummies,
    )

    if duration_col not in cleaned.columns:
        raise ValueError(f"{duration_col} not in cleaned columns")

    if dataset is None:
        dataset = (
            Path(attack_optuna_db).stem if attack_optuna_db is not None else data_name
        )

    survival_config = (
        config
        if resolved_survival_model in config
        else {
            resolved_survival_model: {
                "t0": config.get("t0", 0.35),
                "model": config.get("survival_model_params", {}),
                "plot": config.get("plot", {}),
                "labels": config.get("labels", {}),
            },
        }
    )

    run_results = render_all_survival_model_plots(
        config=survival_config,
        duration_col=duration_col,
        target=target,
        data=cleaned,
        dataset=dataset,
        test_size=test_size,
        folder=output_folder.as_posix(),
        dummy_dict=dummies,
    )

    model_scores = None
    if aux_model is not None:
        runtime_data = run_results["runtime_data"]
        if (
            runtime_data.X_train is None
            or runtime_data.X_test is None
            or runtime_data.y_train is None
            or runtime_data.y_test is None
        ):
            raise ValueError("Runtime survival split unavailable for auxiliary model")
        try:
            model_scores = _evaluate_aux_model(
                model_config=aux_model,
                data_config=runtime_data,
            )
        except Exception as error:
            logger.warning("Aux model evaluation failed: %s", error)

    return {
        "aft_table": run_results["table"],
        "model_scores": model_scores,
        "models": run_results["models"],
    }


survival_parser = create_parser_from_function(survival_main)


if __name__ == "__main__":
    args = survival_parser.parse_args()
    survival_main(**vars(args))
