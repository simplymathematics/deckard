"""Survival-specific experiment orchestration.

This module contains SurvivalExperimentConfig for survival workflows and plotting.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Union, cast

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from lifelines.fitters import RegressionFitter

from ...attack import AttackConfig
from ...data import DataConfig
from ...model import ModelConfig, SurvivalModelConfig
from ...experiment.base import ExperimentConfig


def _lifelines_dataset_loaders() -> dict[str, Any]:
    try:
        from lifelines import datasets as lifelines_datasets
    except Exception:
        return {}

    dataset_names = [
        "c_botulinum_lag_phase",
        "canadian_senators",
        "dd",
        "dfcv",
        "diabetes",
        "g3",
        "gbsg2",
        "holly_molly_polly",
        "kidney_transplant",
        "larynx",
        "leukemia",
        "lung",
        "lymph_node",
        "lymphoma",
        "merrell1955",
        "multicenter_aids_cohort_study",
        "nh4",
        "panel_test",
        "psychiatric_patients",
        "recur",
        "rossi",
        "stanford_heart_transplants",
        "static_test",
        "waltons",
    ]
    return {
        name: getattr(lifelines_datasets, f"load_{name}", None)
        for name in dataset_names
        if hasattr(lifelines_datasets, f"load_{name}")
    }


@dataclass(eq=False, kw_only=True)
class SurvivalExperimentConfig(ExperimentConfig):
    """ExperimentConfig specialization for survival-analysis workflows.

    Required initialization arguments:
    - data: DataConfig
    - model: survival model name (str)
    - target: event label column in runtime data
    - event_col: event/censoring column expected by lifelines fitters
    - duration_col: duration/time column expected by lifelines fitters
    """

    # Required fields
    data: DataConfig
    model = cast(Any, None)
    target: str
    event_col: str
    duration_col: str

    # Optional runtime configuration
    aux_model: Optional[Union[dict[str, Any], "ModelConfig"]] = None
    plots_folder: str = "plots/survival"
    dataset: Optional[str] = None
    model_config: Optional[dict[str, Any]] = None
    calculate_attack_failures: bool = False
    attack_optuna_db: Optional[str] = None
    attack_schema: Optional[Union[str, dict[str, Any]]] = None
    attack_query: Optional[str] = None
    test_size: float = 0.25
    random_state: int = 42
    fillna: Optional[dict[str, Any]] = None
    dummies: Optional[dict[str, Any]] = None
    covariates: Optional[list[str]] = None
    t0: float = 0.35
    survival_model_params: Optional[dict[str, Any]] = None
    plot: Optional[dict[str, Any]] = None
    labels: Optional[dict[str, Any]] = None

    execution_mode: Literal["auto", "native", "auxiliary", "optuna"] = "auto"

    @classmethod
    def infer_execution_mode(
        cls,
        *,
        execution_mode: str = "auto",
        attack_optuna_db: Optional[str] = None,
        attack: Optional[Union[Mapping[str, str], AttackConfig]] = None,
    ) -> str:
        """Resolve which execution mode to use.

        Args:
            execution_mode: Requested mode, or ``"auto"`` to infer from config.
            attack_optuna_db: Optional Optuna database path for attack results.
            attack: Optional attack config or attack mapping.

        Returns:
            The resolved execution mode.

        Raises:
            ValueError: If ``execution_mode`` is not a supported value.
        """
        allowed = {"auto", "native", "auxiliary", "optuna"}
        if execution_mode not in allowed:
            raise ValueError(
                f"execution_mode must be one of {sorted(allowed)}, got {execution_mode!r}",
            )
        if execution_mode != "auto":
            return execution_mode
        if attack_optuna_db is not None:
            return "optuna"
        if attack is not None:
            return "auxiliary"
        return "native"

    @staticmethod
    def fit_aft(
        df: pd.DataFrame,
        event_col: str,
        duration_col: str,
        mtype: str,
        summary_file: Optional[str] = None,
        folder: Optional[str] = None,
        **kwargs,
    ) -> RegressionFitter:
        """Fit a lifelines survival model.

        Args:
            df: Training frame that includes event and duration columns.
            event_col: Event indicator column.
            duration_col: Duration/time column.
            mtype: Lifelines model key (for example, ``"weibull"``).
            summary_file: Optional summary output file name.
            folder: Optional output folder for artifacts.
            **kwargs: Extra fitter keyword arguments.

        Returns:
            The fitted lifelines regression fitter.
        """
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

    @staticmethod
    def survival_probability_calibration(
        model: "RegressionFitter",
        df: pd.DataFrame,
        t0: float,
        ax: Optional[Axes] = None,
        color: str = "red",
        return_curve: bool = False,
        plot: bool = True,
    ) -> Union[
        tuple[Optional[Axes], float, float],
        tuple[Optional[Axes], float, float, pd.DataFrame],
    ]:
        """Compute survival calibration metrics and optionally render a curve.

        Args:
            model: Fitted lifelines regression fitter.
            df: Data used for calibration.
            t0: Time horizon used by calibration.
            ax: Optional axis for plotting.
            color: Calibration curve color.
            return_curve: Whether to return the calibration curve dataframe.
            plot: Whether to render the calibration plot.

        Returns:
            A tuple containing axis, ICI, and E50. If ``return_curve`` is true,
            returns a 4-tuple with the calibration curve dataframe appended.
        """
        config = SurvivalModelConfig(
            model_type="lifelines",
            classifier=False,
            duration_col=cast(Any, model).duration_col,
            event_col=cast(Any, model).event_col,
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

    @staticmethod
    def clean_data_for_aft(
        data: pd.DataFrame,
        covariate_list: list[str],
        target: str = "adv_failure_rate",
        dummy_dict: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Clean and encode tabular data for AFT-style survival fitting.

        Args:
            data: Input dataframe.
            covariate_list: Covariate columns to retain.
            target: Target/event column name.
            dummy_dict: Optional mapping from categorical columns to dummy prefixes.

        Returns:
            A cleaned dataframe ready for lifelines model fitting.
        """
        return SurvivalModelConfig.clean_data_for_aft(
            data,
            covariate_list,
            target,
            dummy_dict,
        )

    @classmethod
    def compute_failures_under_attack(
        cls,
        data: pd.DataFrame,
        attack_config: Optional[AttackConfig] = None,
        benign_metric: str = "accuracy",
    ) -> pd.DataFrame:
        """Derive failure-count columns from attack metrics.

        Args:
            data: Input dataframe containing benign or adversarial metrics.
            attack_config: Optional attack configuration for defaults.
            benign_metric: Column used to compute benign failure rate.

        Returns:
            Dataframe with failure-count columns added when derivable.
        """
        config = cls(
            data=DataConfig(dataset_name="toy"),
            model="weibull",
            target="E",
            duration_col="T",
            event_col="E",
        )
        return config.calculate_failures_under_attack(
            data,
            attack_config,
            benign_metric,
        )

    @staticmethod
    def _require_non_empty_str(name: str, value: Any) -> str:
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _normalize_optional_dict(
        name: str,
        value: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a dict when provided, got {type(value)}")
        return dict(value)

    def _validate_survival_data_model(self) -> None:
        if self.data is None:
            raise ValueError("SurvivalExperimentConfig requires a data config")
        if self.aux_model is not None and self.attack is None:
            raise ValueError(
                "SurvivalExperimentConfig requires an attack config when aux_model is specified",
            )
        if not isinstance(self.data, DataConfig):
            raise TypeError(
                f"Expected data to resolve to DataConfig, got {type(self.data)}",
            )
        self._require_non_empty_str("model", self.model)
        if self.aux_model is not None and not isinstance(self.aux_model, ModelConfig):
            raise TypeError(
                f"Expected aux_model to resolve to ModelConfig, got {type(self.aux_model)}",
            )

    def _before_post_init(self) -> None:
        if self.data is not None and not isinstance(self.data, DataConfig):
            try:
                self.data = self.coerce_component(
                    self.data,
                    DataConfig,
                    default_target="deckard.data.DataConfig",
                )
            except Exception as exc:
                raise TypeError(
                    f"Expected data to resolve to DataConfig, got {type(self.data)}",
                ) from exc
        if self.aux_model is not None and not isinstance(self.aux_model, ModelConfig):
            try:
                self.aux_model = self.coerce_component(
                    self.aux_model,
                    ModelConfig,
                    default_target="deckard.model.ModelConfig",
                )
            except Exception as exc:
                raise TypeError(
                    f"Expected aux_model to resolve to ModelConfig, got {type(self.aux_model)}",
                ) from exc

    def _validate_survival_fields(self) -> None:
        self._require_non_empty_str("duration_col", self.duration_col)
        self._require_non_empty_str("target", self.target)
        self._require_non_empty_str("event_col", self.event_col)

    def __post_init__(self) -> None:
        """Validate and normalize survival experiment configuration fields.

        Raises:
            TypeError: If ``covariates`` is provided but is not list-like.
            ValueError: If required survival fields are missing or invalid.
        """
        self._before_post_init()
        self._validate_survival_data_model()
        self._validate_survival_fields()
        if self.covariates is not None:
            try:
                self.covariates = list(self.covariates)
            except TypeError as exc:
                raise TypeError("covariates must be list-like when provided") from exc

    def _resolve_execution_mode(self) -> str:
        return self.infer_execution_mode(
            execution_mode=self.execution_mode,
            attack_optuna_db=self.attack_optuna_db,
            attack=self.attack,
        )

    def _resolve_covariates(self) -> list[str]:
        covariates = list(self.covariates or [])
        if self.target not in covariates:
            covariates.append(self.target)
        if self.duration_col not in covariates:
            covariates.append(self.duration_col)
        return covariates

    @staticmethod
    def _get_attack_label_column(data: pd.DataFrame) -> Optional[str]:
        """Find the attack label column in the dataframe."""
        for candidate in [
            "attack.alias",
            "attack name",
            "attack_name",
            "attack",
            "attack_alias",
        ]:
            if candidate in data.columns:
                return candidate
        return None

    @staticmethod
    def _infer_attack_kind_from_label(label: Optional[str]) -> Optional[str]:
        """Infer attack kind from a label string."""
        if label is None or (isinstance(label, float) and np.isnan(label)):
            return None
        value = str(label).strip().lower()
        if value == "":
            return None
        if any(token in value for token in ["membership", "member"]):
            return "membership"
        if any(token in value for token in ["attribute", "attr"]):
            return "attribute"
        return "evasion"

    @staticmethod
    def _candidate_attack_metrics_for_kind(
        attack_kind: Optional[str],
    ) -> list[str]:
        """Return candidate metrics for a given attack kind."""
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

    @staticmethod
    def _resolve_attack_size(
        output: pd.DataFrame,
        row_index: Optional[Any] = None,
        attack_config: Optional["AttackConfig"] = None,
    ) -> float:
        """Resolve attack size from output or attack config."""
        if row_index is not None and "attack_size" in output.columns:
            attack_size = output.at[row_index, "attack_size"]
            if not pd.isna(attack_size):
                try:
                    return float(cast(Any, attack_size))
                except (TypeError, ValueError):
                    pass
        if "attack_size" in output.columns and output["attack_size"].notna().all():
            unique_sizes = output["attack_size"].dropna().unique()
            if len(unique_sizes) == 1:
                return float(unique_sizes[0])
        if attack_config is not None:
            return float(attack_config.attack_size)
        return 1.0

    @staticmethod
    def _failure_count_from_metric(
        value: float,
        metric: str,
        attack_size: float,
    ) -> float:
        """Compute failure count from a metric value."""
        failure_rate = value if metric.endswith("_success") else 1 - value
        return attack_size * failure_rate

    def calculate_failures_under_attack(
        self,
        data: pd.DataFrame,
        attack_config: Optional["AttackConfig"] = None,
        benign_metric: str = "accuracy",
    ) -> pd.DataFrame:
        """Compute benign and adversarial failures from attack metrics.

        Args:
            data: Input dataframe containing score columns.
            attack_config: Optional attack configuration for fallback metadata.
            benign_metric: Metric column used for benign failures.

        Returns:
            Dataframe with derived ``ben_failures`` and/or ``adv_failures`` when
            sufficient data exists.
        """
        output = data.copy()
        if benign_metric in output.columns and "ben_failures" not in output.columns:
            if "attack_size" in output.columns:
                attack_sizes = output["attack_size"].fillna(
                    self._resolve_attack_size(
                        output,
                        attack_config=attack_config,
                    ),
                )
            else:
                attack_sizes = pd.Series(
                    self._resolve_attack_size(
                        output,
                        attack_config=attack_config,
                    ),
                    index=output.index,
                    dtype=float,
                )
            output["ben_failures"] = attack_sizes * (1 - output[benign_metric])

        attack_label_col = self._get_attack_label_column(output)
        attack_kind = attack_config.attack_kind if attack_config is not None else None

        if attack_label_col is not None:
            adv_failures = pd.Series(np.nan, index=output.index, dtype=float)
            for row_index, attack_label in output[attack_label_col].items():
                row_kind = (
                    self._infer_attack_kind_from_label(attack_label) or attack_kind
                )
                for metric in self._candidate_attack_metrics_for_kind(row_kind):
                    if metric not in output.columns or pd.isna(
                        output.at[row_index, metric],
                    ):
                        continue
                    value = output.at[row_index, metric]
                    adv_failures.at[row_index] = self._failure_count_from_metric(
                        value=value,
                        metric=metric,
                        attack_size=self._resolve_attack_size(
                            output,
                            row_index=row_index,
                            attack_config=attack_config,
                        ),
                    )
                    break
            if adv_failures.notna().any():
                output["adv_failures"] = adv_failures
                return output

        for metric in self._candidate_attack_metrics_for_kind(attack_kind):
            if metric in output.columns:
                if "attack_size" in output.columns:
                    attack_sizes = output["attack_size"].fillna(
                        self._resolve_attack_size(
                            output,
                            attack_config=attack_config,
                        ),
                    )
                else:
                    attack_sizes = pd.Series(
                        self._resolve_attack_size(
                            output,
                            attack_config=attack_config,
                        ),
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

    def make_survival_model_table(
        self,
        models: Mapping[str, RegressionFitter],
        X_test: pd.DataFrame,
        folder: str = ".",
        t0s: Optional[Mapping[str, float]] = None,
    ) -> pd.DataFrame:
        """Build a survival model comparison table.

        Args:
            models: Mapping of model names to fitted lifelines models.
            X_test: Evaluation dataframe.
            folder: Output folder for the generated CSV table.
            t0s: Optional per-model calibration time horizons.

        Returns:
            A dataframe containing model comparison metrics.
        """
        t0s = t0s or {}
        comparison_data = []

        for model_type, fitter in models.items():
            if fitter is None:
                continue
            t0 = t0s.get(model_type, 0.35)
            row = {"model": model_type, "t0": t0}

            try:
                if hasattr(fitter, "AIC_"):
                    row["AIC"] = fitter.AIC_
            except Exception:
                pass

            try:
                if hasattr(fitter, "AIC_partial_"):
                    row["AIC"] = fitter.AIC_partial_
            except Exception:
                pass

            try:
                if hasattr(fitter, "BIC_"):
                    row["BIC"] = fitter.BIC_
            except Exception:
                pass

            try:
                if hasattr(fitter, "concordance_index_"):
                    row["concordance"] = fitter.concordance_index_
            except Exception:
                pass

            if (
                self.duration_col in X_test.columns
                and self.event_col in X_test.columns
            ):
                try:
                    calibration = self.survival_probability_calibration(
                        model=fitter,
                        df=X_test,
                        t0=t0,
                        plot=False,
                        return_curve=False,
                    )
                    if isinstance(calibration, Mapping):
                        if "ICI" in calibration:
                            row["ICI"] = calibration["ICI"]
                        if "E50" in calibration:
                            row["E50"] = calibration["E50"]
                    elif (
                        isinstance(calibration, (tuple, list))
                        and len(calibration) >= 3
                    ):
                        row["ICI"] = calibration[1]
                        row["E50"] = calibration[2]
                except Exception:
                    pass

            comparison_data.append(row)

        table = pd.DataFrame(comparison_data)
        if not table.empty:
            csv_path = Path(folder) / "aft_comparison.csv"
            table.to_csv(csv_path, index=False)
        return table

    @staticmethod
    def _load_optuna_frame(
        optuna_db: str,
        schema: Optional[Union[str, dict]] = None,
        query: Optional[str] = None,
    ) -> pd.DataFrame:
        from ...layers.compile_results import parse_studies

        frame = parse_studies(optuna_db=optuna_db, schema=schema or {})
        if query is not None:
            frame = frame.query(query)
        if frame.empty:
            raise ValueError(
                f"No attack results found in {optuna_db} after applying filters",
            )
        return frame

    @staticmethod
    def _normalize_data_spec(
        *,
        data_spec: Union[str, dict[str, Any], DataConfig],
        target: str,
    ) -> tuple[Union[dict[str, Any], DataConfig], str]:
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
            return {
                "dataset_name": normalized_data_spec,
                "target": (
                    None
                    if _is_lifelines_dataset_name(normalized_data_spec)
                    else target
                ),
                "classifier": False,
                "stratify": False,
            }, data_name

        if isinstance(data_spec, DataConfig):
            data_spec.dataset_name = _normalize_survival_dataset_name(
                str(data_spec.dataset_name),
            )
            if _is_lifelines_dataset_name(str(data_spec.dataset_name)):
                data_spec.target = None
            return data_spec, data_spec.dataset_name

        if isinstance(data_spec, Mapping):
            spec = dict(data_spec)
            dataset_name_value = spec.get("dataset_name", spec.get("alias"))
            if dataset_name_value is not None:
                normalized_data_spec = _normalize_survival_dataset_name(
                    str(dataset_name_value),
                )
                spec["dataset_name"] = normalized_data_spec
                if _is_lifelines_dataset_name(normalized_data_spec):
                    spec["target"] = None
            data_name = str(spec.get("dataset_name", spec.get("alias", "dataset")))
            return spec, data_name

        raise TypeError(
            f"Unsupported data_spec type {type(data_spec).__name__!r}. "
            "Expected a file path string, DataConfig, or a mapping.",
        )

    @staticmethod
    def _load_data_with_config(
        *,
        data_cfg: DataConfig,
        target: str,
    ) -> pd.DataFrame:
        normalized_cfg, _ = SurvivalExperimentConfig._normalize_data_spec(
            data_spec=data_cfg,
            target=target,
        )
        if isinstance(normalized_cfg, DataConfig):
            data_cfg = normalized_cfg
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
        return loaded_data

    @classmethod
    def run_native_mode(
        cls,
        *,
        data_cfg: DataConfig,
        survival_config: "SurvivalExperimentConfig",
    ) -> tuple[pd.DataFrame, Optional[AttackConfig], Optional[ModelConfig]]:
        """Load data for native lifelines execution.

        Args:
            data_cfg: Data configuration used to load raw features.
            survival_config: Active survival experiment configuration.

        Returns:
            Loaded dataframe, optional attack config, and optional auxiliary model.
        """
        loaded_data = cls._load_data_with_config(
            data_cfg=data_cfg,
            target=survival_config.target,
        )
        return loaded_data, None, None

    @classmethod
    def run_auxiliary_mode(
        cls,
        *,
        data_cfg: DataConfig,
        survival_config: "SurvivalExperimentConfig",
    ) -> tuple[pd.DataFrame, Optional[AttackConfig], Optional[ModelConfig]]:
        """Load data and include attack/auxiliary model context.

        Args:
            data_cfg: Data configuration used to load raw features.
            survival_config: Active survival experiment configuration.

        Returns:
            Loaded dataframe, optional attack config, and optional auxiliary model.
        """
        loaded_data = cls._load_data_with_config(
            data_cfg=data_cfg,
            target=survival_config.target,
        )
        return loaded_data, survival_config.attack, survival_config.aux_model

    @classmethod
    def run_optuna_mode(
        cls,
        *,
        attack_optuna_db: str,
        attack_schema: Optional[Union[str, dict[str, Any]]] = None,
        attack_query: Optional[str] = None,
    ) -> tuple[pd.DataFrame, Optional[AttackConfig], Optional[ModelConfig]]:
        """Load survival-ready data from Optuna attack studies.

        Args:
            attack_optuna_db: Path to the Optuna studies database.
            attack_schema: Optional schema mapping for parsing studies.
            attack_query: Optional pandas query for filtering studies.

        Returns:
            Loaded dataframe, optional attack config, and optional auxiliary model.
        """
        loaded_data = cls._load_optuna_frame(
            optuna_db=attack_optuna_db,
            schema=attack_schema,
            query=attack_query,
        )
        return loaded_data, None, None

    def _resolve_output_folder(self) -> Path:
        output_folder = Path(self.plots_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        return output_folder

    def _build_survival_plot_config(self) -> dict[str, Any]:
        if self.model_config is not None:
            return dict(self.model_config)
        model_params = self._normalize_optional_dict(
            "survival_model_params",
            self.survival_model_params,
        )
        plot_config = self._normalize_optional_dict("plot", self.plot)
        label_config = self._normalize_optional_dict("labels", self.labels)
        return {
            str(self.model): {
                "t0": self.t0,
                "model": model_params,
                "plot": plot_config,
                "labels": label_config,
            },
        }

    def _prepare_loaded_data(self, loaded_data: pd.DataFrame) -> pd.DataFrame:
        loaded_data = loaded_data.copy()
        loaded_data.columns = [str(col).strip() for col in loaded_data.columns]
        if self.duration_col not in loaded_data.columns:
            raise ValueError(
                f"duration_col {self.duration_col!r} not found in data. "
                f"Available columns: {list(loaded_data.columns)}",
            )
        fillna_map = self._normalize_optional_dict("fillna", self.fillna)
        dummy_map = self._normalize_optional_dict("dummies", self.dummies)
        for col, value in fillna_map.items():
            if col not in loaded_data.columns:
                raise ValueError(f"{col} not found in input data")
            loaded_data[col] = loaded_data[col].fillna(value)
        if self.target not in loaded_data.columns:
            raise ValueError(
                f"target {self.target!r} not found in data. "
                f"Available columns: {list(loaded_data.columns)}",
            )
        covariates = self._resolve_covariates()
        cleaned = self.clean_data_for_aft(
            loaded_data,
            covariates,
            target=self.target,
            dummy_dict=dummy_map,
        )
        if self.duration_col not in cleaned.columns:
            raise ValueError(f"{self.duration_col} not in cleaned columns")
        return cleaned

    def __call__(self) -> dict[str, pd.DataFrame | dict[str, RegressionFitter] | Optional[dict[str, float]]]:
        """Run the configured survival experiment workflow.

        Returns:
            Mapping with ``aft_table``, ``model_scores``, and fitted ``models``.
        """
        from .plot import SurvivalSeabornPlotConfigList

        logging.basicConfig(level=logging.INFO)
        matplotlib.rc("font", **{"family": "Times New Roman", "size": 22})

        output_folder = self._resolve_output_folder()
        resolved_mode = self._resolve_execution_mode()
        dummy_map = self._normalize_optional_dict("dummies", self.dummies)

        if resolved_mode == "optuna":
            if self.attack_optuna_db is None:
                raise ValueError(
                    "attack_optuna_db is required for execution_mode='optuna'",
                )
            loaded_data, attack_cfg, aux_model = self.run_optuna_mode(
                attack_optuna_db=self.attack_optuna_db,
                attack_schema=self.attack_schema,
                attack_query=self.attack_query,
            )
            data_name = Path(self.attack_optuna_db).stem
        elif resolved_mode == "auxiliary":
            if self.attack is None:
                raise ValueError("attack is required for execution_mode='auxiliary'")
            loaded_data, attack_cfg, aux_model = self.run_auxiliary_mode(
                data_cfg=self.data,
                survival_config=self,
            )
            data_name = getattr(self.data, "dataset_name", None) or "data"
        else:
            loaded_data, attack_cfg, aux_model = self.run_native_mode(
                data_cfg=self.data,
                survival_config=self,
            )
            data_name = getattr(self.data, "dataset_name", None) or "data"

        if self.calculate_attack_failures or self.target in {
            "ben_failures",
            "adv_failures",
        }:
            loaded_data = self.compute_failures_under_attack(
                loaded_data,
                attack_config=attack_cfg,
                benign_metric="accuracy",
            )

        cleaned = self._prepare_loaded_data(loaded_data)
        dataset = self.dataset or data_name
        survival_config = self._build_survival_plot_config()

        plot_config_list = SurvivalSeabornPlotConfigList()
        run_results = plot_config_list(
            model_config=survival_config,
            data=cleaned,
            survival_config=self,
            dataset=dataset,
            test_size=self.test_size,
            folder=output_folder.as_posix(),
            dummy_dict=dummy_map,
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
            model_scores = aux_model(runtime_data)

        return {
            "aft_table": run_results["table"],
            "model_scores": model_scores,
            "models": run_results["models"],
        }
