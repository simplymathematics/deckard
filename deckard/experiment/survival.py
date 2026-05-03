"""Survival-specific experiment orchestration.

This module contains :class:`SurvivalExperimentConfig`, separated from the core
experiment module so survival workflows can be treated as an optional extension.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..data import DataConfig
from .base import ExperimentConfig
from ..model import ModelConfig
from ..attack import AttackConfig


class SurvivalExperimentConfig(ExperimentConfig):
    """ExperimentConfig specialization for survival-analysis workflows."""

    """Experiment configuration tailored for survival analyses.

    This config enforces that both `data` and `model` are provided and valid,
    while allowing survival-specific settings to be carried alongside standard
    experiment settings.
    """

    survival_model = "weibull"
    duration_col = "T"
    event_col = "E"

    def __post_init__(self):
        super().__post_init__()
        if self.data is None:
            raise ValueError("SurvivalExperimentConfig requires a data config")
        if self.attack is not None and self.model is None:
            raise ValueError(
                "SurvivalExperimentConfig requires a model config when an attack is specified",
            )
        if not isinstance(self.data, DataConfig):
            raise TypeError(
                f"Expected data to resolve to DataConfig, got {type(self.data)}",
            )
        if self.model is not None and not isinstance(self.model, ModelConfig):
            raise TypeError(
                f"Expected model to resolve to ModelConfig, got {type(self.model)}",
            )
        if self.duration_col in [None, ""]:
            raise ValueError(
                "duration_col must be provided for survival experiments",
            )

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
                return float(attack_size)
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
        """Optionally derive ben/adv failure counts from attack-specific accuracy metrics."""
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
        models: dict,
        dataset: Optional[str],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        folder: str = ".",
        t0s: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Build comparison table with AIC/BIC/Concordance/ICI/E50 columns."""
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
                    from deckard.layers.survival import (
                        survival_probability_calibration,
                    )

                    calibration = survival_probability_calibration(
                        fitter=fitter,
                        X_test=X_test,
                        duration_col=self.duration_col,
                        event_col=self.event_col,
                        t0=t0,
                    )
                    if calibration is not None:
                        if "ICI" in calibration:
                            row["ICI"] = calibration["ICI"]
                        if "E50" in calibration:
                            row["E50"] = calibration["E50"]
                except Exception:
                    pass

            comparison_data.append(row)

        table = pd.DataFrame(comparison_data)
        if not table.empty:
            csv_path = Path(folder) / "aft_comparison.csv"
            table.to_csv(csv_path, index=False)
        return table
