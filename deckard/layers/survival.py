import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import matplotlib
import numpy as np
import pandas as pd
import yaml
from lifelines.fitters import RegressionFitter

from ..attack import AttackConfig
from ..data import DataConfig
from ..data.base import _lifelines_dataset_loaders
from ..experiment import SurvivalExperimentConfig
from ..model.survival import SurvivalModelConfig
from .compile_results import parse_studies
from ..plot.survival import (
    SurvivalSeabornPlotterConfig,
    SurvivalSeabornPlotConfigList,
)
from ..utils import create_parser_from_function

logger = logging.getLogger(__name__)

__all__ = [
    "survival_main",
    "run_survival_model_experiment",
    "render_all_survival_model_plots",
    "survival_parser",
]


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
    config = SurvivalModelConfig(
        model_type="lifelines",
        classifier=False,
        survival_model=mtype,
        duration_col=duration_col,
        event_col=event_col,
    )
    return config.fit_aft(
        df=df,
        summary_file=summary_file,
        folder=folder,
        **kwargs,
    )


def survival_probability_calibration(
    model: RegressionFitter,
    df: pd.DataFrame,
    t0: float,
    ax=None,
    color: str = "red",
    return_curve: bool = False,
    plot: bool = True,
) -> Union[tuple[Any, float, float], tuple[Any, float, float, pd.DataFrame]]:
    """Compute survival calibration metrics and optionally render a calibration curve."""
    config = SurvivalModelConfig(
        model_type="lifelines",
        classifier=False,
        duration_col=model.duration_col,
        event_col=model.event_col,
        t0=t0,
    )
    return config.survival_probability_calibration(
        model=model,
        df=df,
        ax=ax,
        color=color,
        return_curve=return_curve,
        plot=plot,
    )


def clean_data_for_aft(
    data: pd.DataFrame,
    covariate_list: list,
    target: str = "adv_failure_rate",
    dummy_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """Clean and encode tabular data for AFT-style survival fitting."""
    return SurvivalModelConfig.clean_data_for_aft(
        data,
        covariate_list,
        target,
        dummy_dict,
    )


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
    plotter = SurvivalSeabornPlotterConfig()
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

    aft_plot = plotter.plot_aft(
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

    qq_plot = plotter.plot_qq(
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
                survival_probability_calibration(
                    model,
                    frame,
                    t0=cutoff,
                    return_curve=True,
                    plot=False,
                )[3].assign(dataset="train"),
                *(
                    [
                        survival_probability_calibration(
                            model,
                            frame_test,
                            t0=cutoff,
                            return_curve=True,
                            plot=False,
                        )[3].assign(dataset="test"),
                    ]
                    if frame_test is not None
                    else []
                ),
            ],
            ignore_index=True,
        ),
        folder=folder,
    )
    plots.append(qq_plot)

    if plot_dict.get("summary_plot") is not None:
        summary_plot = plotter.plot_summary(
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
        )
        plots.append(summary_plot)

    for partial_effect_dict in partial_effect_list:
        effect_config = dict(partial_effect_dict)
        file = effect_config.pop("file", "partial_effects.pdf")
        partial_effect_plot = plotter.plot_partial_effects(
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
    plot_config_list = SurvivalSeabornPlotConfigList()
    return plot_config_list.orchestrate_survival_models(
        model_config=config,
        data=data,
        duration_col=duration_col,
        target=target,
        dataset=dataset,
        test_size=test_size,
        folder=folder,
        dummy_dict=dummy_dict or {},
    )


def _load_optuna_frame(
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


def calculate_failures_under_attack(
    data: pd.DataFrame,
    attack_config: Optional[AttackConfig] = None,
    benign_metric: str = "accuracy",
) -> pd.DataFrame:
    """Optionally derive ben/adv failure counts from attack-specific accuracy metrics."""
    config = SurvivalExperimentConfig(data=DataConfig(dataset_name="toy"))
    return config.calculate_failures_under_attack(
        data,
        attack_config,
        benign_metric,
    )


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
    cfg: Any = None,
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

    if isinstance(cfg, Mapping):
        cfg_block = cfg.get("survival", cfg)
        if isinstance(cfg_block, Mapping):
            config.update(dict(cfg_block))

    if (data == "lung" or isinstance(data, Mapping)) and "data" in config:
        data = config.pop("data")
    if (model == "weibull" or isinstance(model, Mapping)) and "model" in config:
        model = config.pop("model")
    if plots_folder == "plots/survival" and "plots_folder" in config:
        plots_folder = str(config.pop("plots_folder"))
    if target == "E" and "target" in config:
        target = str(config.pop("target"))
    if duration_col == "T" and "duration_col" in config:
        duration_col = str(config.pop("duration_col"))
    if dataset is None and "dataset" in config:
        dataset = config.pop("dataset")
    if model_config is None and "model_config" in config:
        model_config = config.pop("model_config")
    if survival_model is None and "survival_model" in config:
        survival_model = config.pop("survival_model")
    if attack is None and "attack" in config:
        attack = config.pop("attack")
    if data_file is None and "data_file" in config:
        data_file = config.pop("data_file")
    if attack_optuna_db is None and "attack_optuna_db" in config:
        attack_optuna_db = config.pop("attack_optuna_db")
    if attack_schema is None and "attack_schema" in config:
        attack_schema = config.pop("attack_schema")
    if attack_query is None and "attack_query" in config:
        attack_query = config.pop("attack_query")
    if not calculate_attack_failures and "calculate_attack_failures" in config:
        calculate_attack_failures = bool(
            config.pop("calculate_attack_failures"),
        )

    output_folder = Path(plots_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    test_size = float(config.pop("test_size", 0.25))
    random_state = int(config.pop("random_state", 42))
    fillna = dict(config.pop("fillna", {}))
    dummies = dict(config.pop("dummies", {}))
    covariates = list(config.pop("covariates", [duration_col, target]))
    data_spec: Union[str, dict[str, Any], DataConfig] = (
        data_file if data_file else data
    )
    lifelines_dataset_names = set(_lifelines_dataset_loaders().keys())

    def _is_lifelines_dataset_name(name: str) -> bool:
        if name in lifelines_dataset_names:
            return True
        if name.startswith("lifelines."):
            return name.split("lifelines.", 1)[1] in lifelines_dataset_names
        if name.startswith("lifelines_"):
            return name.split("lifelines_", 1)[1] in lifelines_dataset_names
        return False

    def _normalize_survival_dataset_name(name: str) -> str:
        if name in lifelines_dataset_names:
            return f"lifelines.{name}"
        return name

    if isinstance(data_spec, str):
        normalized_data_spec = _normalize_survival_dataset_name(data_spec)
        data_name = (
            Path(normalized_data_spec).stem
            if Path(normalized_data_spec).suffix
            else normalized_data_spec
        )
        data_spec = {
            "dataset_name": normalized_data_spec,
            "target": (
                None
                if _is_lifelines_dataset_name(normalized_data_spec)
                else target
            ),
            "classifier": False,
            "stratify": False,
            "test_size": test_size,
            "random_state": random_state,
        }
    elif isinstance(data_spec, DataConfig):
        data_spec.dataset_name = _normalize_survival_dataset_name(
            str(data_spec.dataset_name),
        )
        if _is_lifelines_dataset_name(str(data_spec.dataset_name)):
            data_spec.target = None
        data_name = data_spec.dataset_name
    elif isinstance(data_spec, Mapping):
        data_spec = dict(data_spec)
        dataset_name_value = data_spec.get(
            "dataset_name",
            data_spec.get("alias"),
        )
        if dataset_name_value is not None:
            normalized_data_spec = _normalize_survival_dataset_name(
                str(dataset_name_value),
            )
            data_spec["dataset_name"] = normalized_data_spec
            if _is_lifelines_dataset_name(normalized_data_spec):
                data_spec["target"] = None
        data_name = str(
            data_spec.get("dataset_name", data_spec.get("alias", "dataset")),
        )
    else:
        data_name = Path(str(data_spec)).stem

    resolved_survival_model = survival_model
    if resolved_survival_model is None:
        if isinstance(model, str):
            resolved_survival_model = model
        elif isinstance(model, Mapping):
            explicit_model_name = (
                model.get("survival_model")
                or model.get("model")
                or model.get("model_type")
                or model.get("alias")
            )
            if isinstance(explicit_model_name, str):
                resolved_survival_model = explicit_model_name
    if resolved_survival_model is None and isinstance(cfg, Mapping):
        cfg_model = cfg.get("model")
        if isinstance(cfg_model, str):
            resolved_survival_model = cfg_model
        elif isinstance(cfg.get("survival"), Mapping):
            nested_model = cfg["survival"].get("model")
            if isinstance(nested_model, str):
                resolved_survival_model = nested_model
    if resolved_survival_model is None:
        resolved_survival_model = "weibull"

    experiment = None
    attack_cfg: Optional[AttackConfig] = (
        attack if isinstance(attack, AttackConfig) else None
    )
    aux_model = None
    if attack_optuna_db is not None:
        loaded_data = _load_optuna_frame(
            optuna_db=attack_optuna_db,
            schema=attack_schema,
            query=attack_query,
        )
    else:
        aux_model_spec = None
        if attack is not None:
            aux_model_spec = model_config if model_config is not None else model
        experiment = SurvivalExperimentConfig(
            data=data_spec,
            model=aux_model_spec,
            attack=attack,
            survival_model=resolved_survival_model,
            duration_col=duration_col,
            event_col=target,
            classifier=False,
            library="sklearn",
        )
        attack_cfg = experiment.attack
        aux_model = experiment.model if experiment.attack is not None else None

        data_cfg = experiment.data
        if data_cfg.X is None:
            data_cfg._load_data()
        loaded_frame = data_cfg.X
        if loaded_frame is None:
            raise ValueError(
                "DataConfig did not load features for survival experiment",
            )
        loaded_data = (
            loaded_frame.to_frame().copy()
            if isinstance(loaded_frame, pd.Series)
            else pd.DataFrame(loaded_frame).copy()
        )
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
            Path(attack_optuna_db).stem
            if attack_optuna_db is not None
            else data_name
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
            raise ValueError(
                "Runtime survival split unavailable for auxiliary model",
            )
        try:
            model_scores = aux_model(runtime_data)
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
