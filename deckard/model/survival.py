"""Survival analysis model configuration and orchestration.

This module provides SurvivalModelConfig, a specialized ModelConfig for
fitting and scoring lifelines survival models with support for AFT-style
fitters, calibration metrics, and model comparison tables.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from .base import ModelConfig
from ..utils import save_data

ScorerDictConfig = Any

logger = logging.getLogger(__name__)

AFT_MODEL_TYPES = {
    "weibull": WeibullAFTFitter,
    "log_normal": LogNormalAFTFitter,
    "log_logistic": LogLogisticAFTFitter,
    "cox": CoxPHFitter,
    "aalen": AalenAdditiveFitter,
    "gamma": GeneralizedGammaRegressionFitter,
    "exponential": PiecewiseExponentialRegressionFitter,
}


@dataclass(eq=False)
class SurvivalModelConfig(ModelConfig):
    """Configuration for survival analysis models using lifelines.

    Extends ModelConfig to support AFT (Accelerated Failure Time) survival
    models. Handles fitting, calibration scoring, and model comparison table
    generation.

    Attributes
    ----------
    duration_col : str
        Column name for duration/time values.
    event_col : str
        Column name for event indicators.
    survival_model : str
        Type of survival model (e.g., "weibull", "cox").
    t0 : float
        Time point for calibration scoring.
    """
    classifier = False # Survival Models are always regression models. Auxilary models may not be.
    duration_col: str = "T"
    event_col: str = "E"
    survival_model: str = "weibull"
    t0: float = 0.35
    
    

    def _initialize_runtime_fields(self) -> None:
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}
        for attr in [
            "training_time",
            "prediction_time",
            "training_prediction_time",
            "training_score_time",
            "prediction_score_time",
            "defense_application_time",
            "training_n",
            "prediction_n",
            "training_predictions",
            "predictions",
            "training_probabilities",
            "probabilities",
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, None)

    def _initialize_target(self) -> None:
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.model.SurvivalModelConfig"


    def __post_init__(self):
        """Initialize SurvivalModelConfig without loading a model through Hydra."""
        # Survival models are always regression models regardless of what was passed.
        self.classifier = False
        # Skip ModelConfig's __post_init__ which tries to load a model.
        self._initialize_runtime_fields()
        self._initialize_target()

    @staticmethod
    def _initialize_aft_fitter(mtype: str, kwargs: dict) -> RegressionFitter:
        """Initialize a lifelines AFT fitter with appropriate defaults."""
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

    @staticmethod
    def _ccl(probabilities: np.ndarray) -> np.ndarray:
        """Complementary log-log transformation for calibration."""
        return np.log(-np.log(1 - probabilities))

    def fit_aft(
        self,
        df: pd.DataFrame,
        summary_file: Optional[str] = None,
        folder: Optional[str] = None,
        **kwargs,
    ) -> RegressionFitter:
        """Fit a survival model and optionally persist its summary."""
        if self.duration_col not in df.columns:
            raise ValueError(f"Column {self.duration_col} not found in data")
        if self.event_col is not None and self.event_col not in df.columns:
            raise ValueError(f"Column {self.event_col} not found in data")

        aft = self._initialize_aft_fitter(
            mtype=self.survival_model,
            kwargs=kwargs,
        )
        fit_kwargs = {
            "duration_col": self.duration_col,
            "event_col": self.event_col,
        }
        if self.survival_model != "aalen":
            start = df[self.duration_col].min()
            end = df[self.duration_col].max()
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

        self._model = aft
        return aft

    def _score(
        self,
        y_true: pd.DataFrame,
        y_pred: RegressionFitter,
        mode: str = "test",
        **kwargs,
    ) -> dict:
        """Compute survival model scores (calibration metrics).

        For survival models, y_pred is the fitted fitter and y_true contains
        duration and event columns.

        Parameters
        ----------
        y_true : pd.DataFrame
            DataFrame with duration_col and event_col.
        y_pred : RegressionFitter
            Fitted survival model.

        Returns
        -------
        dict
            Dictionary with calibration metrics (ici, e50, concordance).
        """
        if getattr(self, "scorer", None) is not None:
            return self.scorer(
                y_true=y_true,
                y_pred=y_pred,
                mode=mode,
                data=y_true,
                **kwargs,
            )

        scores = {}

        # Compute concordance if available
        try:
            if hasattr(y_pred, "concordance_index_"):
                scores["concordance"] = y_pred.concordance_index_
        except Exception as e:
            logger.warning(f"Could not compute concordance: {e}")

        # Compute calibration metrics
        if self.duration_col in y_true.columns and self.event_col in y_true.columns:
            try:
                _, ici, e50 = self.survival_probability_calibration(
                    model=y_pred,
                    df=y_true,
                    plot=False,
                )
                if not np.isnan(ici):
                    scores["ici"] = ici
                if not np.isnan(e50):
                    scores["e50"] = e50
            except Exception as e:
                logger.warning(f"Could not compute calibration metrics: {e}")

        return scores

    def survival_probability_calibration(
        self,
        model: RegressionFitter,
        df: pd.DataFrame,
        ax=None,
        color: str = "red",
        return_curve: bool = False,
        plot: bool = True,
    ) -> Union[
        tuple[Any, float, float],
        tuple[Any, float, float, pd.DataFrame],
    ]:
        """Compute survival calibration metrics and optionally render curve."""
        if plot:
            if not ax:
                _, ax = plt.subplots()

        duration_col = model.duration_col
        event_col = model.event_col
        calibration_df = df.copy()
        for col in calibration_df.columns:
            calibration_df[col] = pd.to_numeric(
                calibration_df[col],
                errors="raise",
            )
        calibration_df = calibration_df.dropna()

        predictions_at_t0 = np.clip(
            1
            - model.predict_survival_function(
                calibration_df,
                times=[self.t0],
            ).T.squeeze(),
            1e-10,
            1 - 1e-10,
        )

        t0_tag = str(self.t0).replace(".", "_")
        predictor_col = f"ccl_at_{t0_tag}"
        prediction_df = pd.DataFrame(
            {
                predictor_col: self._ccl(predictions_at_t0),
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
                logger.error(
                    "Could not fit CRC model for calibration: %s",
                    error,
                )
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
                pd.DataFrame({predictor_col: self._ccl(x)}),
                times=[self.t0],
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
                (
                    1
                    - crc.predict_survival_function(
                        prediction_df,
                        times=[self.t0],
                    )
                ).T.squeeze()
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

    @staticmethod
    def clean_data_for_aft(
        data: pd.DataFrame,
        covariate_list: list,
        target: str = "adv_failure_rate",
        dummy_dict: Optional[dict] = None,
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
            available_dummy_cols = [
                c for c in dummy_dict.keys() if c in subset.columns
            ]
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
            object_cols = [
                col for col in cleaned.columns if cleaned[col].dtype == "object"
            ]
            if len(object_cols) > 0:
                dummies = pd.get_dummies(
                    cleaned[object_cols],
                    prefix="",
                    prefix_sep="",
                )
                cleaned = cleaned.drop(columns=object_cols)
                cleaned = pd.concat([cleaned, dummies], axis=1)
            cleaned = cleaned.astype(float)

        cleaned = cleaned.dropna(axis=0, how="any")
        if target not in cleaned.columns:
            raise ValueError(f"Target {target} not in cleaned dataframe")
        return cleaned

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
            t0_for_model = t0s.get(model_type, self.t0)
            row = {"model": model_type, "t0": t0_for_model}

            # Compute AIC and BIC if available
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

            # Optionally compute ICI/E50 if needed
            if (
                self.duration_col in X_test.columns
                and self.event_col in X_test.columns
            ):
                try:
                    old_t0 = self.t0
                    self.t0 = t0_for_model
                    _, ici, e50 = self.survival_probability_calibration(
                        model=fitter,
                        df=X_test,
                        plot=False,
                    )
                    self.t0 = old_t0
                    if not np.isnan(ici):
                        row["ICI"] = ici
                    if not np.isnan(e50):
                        row["E50"] = e50
                except Exception:
                    pass

            comparison_data.append(row)

        table = pd.DataFrame(comparison_data)
        if not table.empty:
            csv_path = Path(folder) / "aft_comparison.csv"
            table.to_csv(csv_path, index=False)
        return table
