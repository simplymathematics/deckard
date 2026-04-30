import inspect
import time
from typing import Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from fairlearn.metrics import (
    MetricFrame,
    false_negative_rate,
    false_positive_rate,
    true_negative_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from art.config import ART_NUMPY_DTYPE

from .base import ModelConfig, logger
from .defend import DefenseConfig

from ..data.fairness import FairnessDataConfig
from ..utils import load_class, resolve_class


class _FairnessBehaviorMixin:
    """
    A model configuration that extends ModelConfig to support fairness-aware evaluation.

    Trains normally on X_train, y_train, but uses group information from X_test, y_test
    for fairness-aware scoring across demographic groups.

    Attributes:
    -----------
    fairness_data : FairnessDataConfig or None
        Configuration containing group information and fairness metrics.
    """

    def _validate_sensitive_series(self, sensitive, context: str):
        if sensitive is None:
            return None
        sensitive_series = pd.Series(sensitive)
        if len(sensitive_series) == 0:
            raise ValueError(f"Sensitive features are empty during {context}")
        if sensitive_series.dropna().empty:
            raise ValueError(f"Sensitive features are all null during {context}")
        if sensitive_series.astype(str).str.strip().eq("").all():
            raise ValueError(f"Sensitive features are blank during {context}")
        return sensitive_series

    def _resolve_sensitive_features_for_batch(self, batch):
        if self.data is None:
            return None

        n_rows = len(batch)
        batch_index = getattr(batch, "index", None)

        candidates = [
            getattr(self.data, "sensitive_train_", None),
            getattr(self.data, "sensitive_test_", None),
            getattr(self.data, "sensitive_all_", None),
        ]

        positional_matches = []
        for sensitive in candidates:
            if sensitive is None:
                continue
            sensitive_series = self._validate_sensitive_series(sensitive, "runtime")
            if sensitive_series is None:
                continue
            if len(sensitive_series) == n_rows:
                if batch_index is not None:
                    try:
                        aligned = sensitive_series.reindex(batch_index)
                        if len(aligned) == n_rows and aligned.notna().all():
                            return aligned.reset_index(drop=True)
                    except Exception:
                        pass
                positional_matches.append(sensitive_series.reset_index(drop=True))

        if len(positional_matches) == 1:
            return positional_matches[0]
        return None

    def _method_accepts_sensitive_features(self, method) -> bool:
        try:
            params = inspect.signature(method).parameters
            if "sensitive_features" in params:
                return True
            # fairlearn reductions (e.g., GridSearch.fit) accept sensitive features via **kwargs
            return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            return False

    def _call_with_optional_sensitive(self, method, X, sensitive):
        if sensitive is not None and self._method_accepts_sensitive_features(method):
            return method(X, sensitive_features=sensitive)
        return method(X)

    def _fit_defended_estimator(self, defended_estimator, data):
        """Fit a defended estimator when defense application requires training."""
        if data is None or not hasattr(defended_estimator, "fit"):
            return defended_estimator

        sensitive = self._resolve_sensitive_features_for_batch(data.y_train)
        fit_method = defended_estimator.fit
        if sensitive is not None and self._method_accepts_sensitive_features(
            fit_method,
        ):
            fit_method(data.X_train, data.y_train, sensitive_features=sensitive)
        else:
            fit_method(data.X_train, data.y_train)
        return defended_estimator

    def _resolve_fairness_defense_spec(self):
        """Resolve defense_name/defense_params from either model or defense config shapes."""
        if hasattr(self, "defense_name"):
            return getattr(self, "defense_name", None), dict(
                getattr(self, "defense_params", {}) or {},
            )
        defense_obj = getattr(self, "defense", None)
        if defense_obj is not None:
            return getattr(defense_obj, "defense_name", None), dict(
                getattr(defense_obj, "defense_params", {}) or {},
            )
        return None, {}

    def _apply_defense(self, data):
        """Apply fairlearn defenses when configured, otherwise defer to base behavior."""
        defense_name, defense_params = self._resolve_fairness_defense_spec()
        if not defense_name or not defense_name.startswith("fairlearn."):
            return ModelConfig._apply_defense(self, data)

        if self._model is None:
            raise ValueError(
                "Fairness model must have a fitted estimator before applying defense",
            )

        module_name, class_name = defense_name.rsplit(".", 1)
        try:
            defense_class = resolve_class(defense_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not import defense class {class_name} from {module_name}",
            ) from e

        constraints = None
        if "constraints" in defense_params:
            constraints_raw = defense_params.pop("constraints")
            if isinstance(constraints_raw, str) and "." in constraints_raw:
                constraints = load_class(constraints_raw)
            elif isinstance(constraints_raw, dict):
                constraints_cfg = dict(constraints_raw)
                c_target = constraints_cfg.pop("_target_", None)
                if c_target is None:
                    raise ValueError("constraints dict must include '_target_'")
                constraints = load_class(c_target, **constraints_cfg)
            else:
                constraints = constraints_raw

        fairlearn_submodule = module_name.split(".")[1]
        base_estimator = self.get_model()

        start = time.process_time()
        if fairlearn_submodule == "reductions":
            if constraints is None:
                raise ValueError(
                    "fairlearn.reductions defenses require a 'constraints' key in defense parameters",
                )
            defended_estimator = defense_class(
                base_estimator,
                constraints,
                **defense_params,
            )
        elif fairlearn_submodule == "postprocessing":
            if constraints is not None:
                defended_estimator = defense_class(
                    estimator=base_estimator,
                    constraints=constraints,
                    **defense_params,
                )
            else:
                defended_estimator = defense_class(
                    estimator=base_estimator,
                    **defense_params,
                )
        elif fairlearn_submodule == "adversarial":
            defended_estimator = defense_class(**defense_params)
        else:
            raise NotImplementedError(
                f"Fairlearn submodule '{fairlearn_submodule}' is not supported. "
                "Expected one of: reductions, postprocessing, adversarial.",
            )

        self._apply_fit = True
        if self._apply_fit:
            defended_estimator = self._fit_defended_estimator(defended_estimator, data)
        end = time.process_time()
        self.defense_application_time = end - start
        return defended_estimator

    def _train(self, X: pd.DataFrame, y: pd.Series):
        if self._model is None:
            raise ValueError("Model not initialized")
        start_time = time.process_time()
        assert hasattr(self._model, "fit"), "Model does not have a fit method"
        fit_params = {} if not hasattr(self, "fit_params") else self.fit_params
        sensitive = self._resolve_sensitive_features_for_batch(y)
        if (
            sensitive is not None
            and self._method_accepts_sensitive_features(self._model.fit)
            and "sensitive_features" not in fit_params
        ):
            fit_params = {**fit_params, "sensitive_features": sensitive}
        self._model.fit(X, y, **fit_params)
        end_time = time.process_time()
        self.training_time = end_time - start_time
        self.training_n = len(y)
        logger.info(f"Model trained in {self.training_time:.2f} seconds")

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise ValueError("Model not initialized")
        sensitive = self._resolve_sensitive_features_for_batch(X)
        try:
            y_pred = self._call_with_optional_sensitive(
                self._model.predict,
                X,
                sensitive,
            )
        except TypeError as e:
            if "loop of ufunc does not support argument" in str(
                e,
            ) or "can't convert" in str(e):
                X_array = np.array(X, dtype=ART_NUMPY_DTYPE)
                y_pred = self._call_with_optional_sensitive(
                    self._model.predict,
                    X_array,
                    sensitive,
                )
            else:
                raise e
        return y_pred

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model not initialized")
        if not self.probability:
            raise ValueError("Model does not support probability predictions")
        sensitive = self._resolve_sensitive_features_for_batch(X)
        return self._call_with_optional_sensitive(
            self._model.predict_proba,
            X,
            sensitive,
        )

    def _resolve_sensitive_features(self, y_true: pd.Series):
        if self.data is None:
            return None

        y_true_series = pd.Series(y_true)
        y_true_n = len(y_true_series)
        y_index = getattr(y_true, "index", None)
        candidates = [
            getattr(self.data, "sensitive_test_", None),
            getattr(self.data, "sensitive_train_", None),
            getattr(self.data, "sensitive_all_", None),
        ]
        positional_matches = []
        for sensitive in candidates:
            if sensitive is None:
                continue
            sensitive_series = self._validate_sensitive_series(
                sensitive,
                "fairness scoring",
            )
            if sensitive_series is None:
                continue
            if len(sensitive_series) == y_true_n:
                if y_index is not None:
                    try:
                        aligned = sensitive_series.reindex(y_index)
                        if len(aligned) == y_true_n and aligned.notna().all():
                            return aligned.reset_index(drop=True)
                    except Exception:
                        pass
                positional_matches.append(sensitive_series.reset_index(drop=True))
        if len(positional_matches) == 1:
            return positional_matches[0]
        return None

    def _compute_group_fairness_scores(self, y_true, y_pred) -> dict:
        sensitive = self._resolve_sensitive_features(y_true)
        if sensitive is None:
            if self.data is not None:
                raise ValueError(
                    "Sensitive features are required for fairness scoring and cannot be empty at runtime",
                )
            return {}

        y_true_series = pd.Series(y_true).reset_index(drop=True)

        def _normalize_pred_labels(y_pred_raw, y_true_ref):
            y_pred_arr = np.asarray(y_pred_raw)
            if y_pred_arr.ndim <= 1:
                return pd.Series(y_pred_arr).reset_index(drop=True)
            if y_pred_arr.ndim != 2:
                raise ValueError(
                    f"Unsupported prediction shape for fairness scoring: {y_pred_arr.shape}",
                )

            classes = np.unique(
                np.asarray(y_true_ref)[~pd.isna(np.asarray(y_true_ref))],
            )

            if y_pred_arr.shape[1] == 1:
                binary_scores = y_pred_arr.reshape(-1)
                threshold = 0.5
                if np.nanmin(binary_scores) < 0.0 or np.nanmax(binary_scores) > 1.0:
                    threshold = 0.0
                if len(classes) == 2 and np.issubdtype(
                    np.asarray(classes).dtype,
                    np.number,
                ):
                    sorted_classes = np.sort(np.asarray(classes, dtype=float))
                    low_label, high_label = sorted_classes[0], sorted_classes[1]
                    labels = np.where(binary_scores >= threshold, high_label, low_label)
                else:
                    labels = (binary_scores >= threshold).astype(int)
                return pd.Series(labels).reset_index(drop=True)

            # Multi-class: argmax gives indices, map to actual class labels
            class_indices = np.argsort(classes)
            sorted_classes = classes[class_indices]
            pred_indices = np.argmax(y_pred_arr, axis=1)
            labels = sorted_classes[pred_indices]
            return pd.Series(labels).reset_index(drop=True)

        y_pred_series = _normalize_pred_labels(y_pred, y_true_series)
        # Determine positive label BEFORE alignment so it stays consistent
        positive_label = sorted(y_true_series.dropna().unique())[-1]

        if self.classifier:
            metric_frame = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "precision": lambda yt, yp: precision_score(
                        yt,
                        yp,
                        average="weighted",
                        zero_division=0,
                    ),
                    "recall": lambda yt, yp: recall_score(
                        yt,
                        yp,
                        average="weighted",
                        zero_division=0,
                    ),
                    "f1-score": lambda yt, yp: f1_score(
                        yt,
                        yp,
                        average="weighted",
                        zero_division=0,
                    ),
                },
                y_true=y_true_series,
                y_pred=y_pred_series,
                sensitive_features=sensitive,
            )
            by_group = metric_frame.by_group.to_dict(orient="index")
            group_scores = {
                f"{group}_{metric}": float(value)
                for group, metrics in by_group.items()
                for metric, value in metrics.items()
            }

            score_df = pd.DataFrame(
                {
                    "y_true": y_true_series,
                    "y_pred": y_pred_series,
                    "sensitive": pd.Series(sensitive).reset_index(drop=True),
                },
            )

            # Calculate sensitive feature importance based on accuracy variation across groups
            group_accuracies = []
            for group, group_df in score_df.groupby("sensitive", sort=False):
                group_acc = float((group_df["y_true"] == group_df["y_pred"]).mean())
                group_accuracies.append(group_acc)

            if group_accuracies:
                acc_arr = np.asarray(group_accuracies, dtype=float)
                acc_min = float(np.min(acc_arr))
                acc_max = float(np.max(acc_arr))
                acc_diff = acc_max - acc_min
                acc_std = float(np.std(acc_arr))
                acc_ratio = (
                    acc_min / acc_max
                    if not np.isclose(acc_max, 0.0)
                    else (1.0 if np.isclose(acc_min, 0.0) else 0.0)
                )
                group_scores["sensitive_feature_accuracy_difference"] = acc_diff
                group_scores["sensitive_feature_accuracy_std"] = acc_std
                group_scores["sensitive_feature_accuracy_ratio"] = acc_ratio

            return group_scores
        # if not classifier
        metric_frame = MetricFrame(
            metrics={
                "mse": lambda yt, yp: float(
                    np.mean((np.asarray(yt) - np.asarray(yp)) ** 2),
                ),
                "rmse": lambda yt, yp: float(
                    np.sqrt(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2)),
                ),
                "mae": lambda yt, yp: float(
                    np.mean(np.abs(np.asarray(yt) - np.asarray(yp))),
                ),
            },
            y_true=y_true_series,
            y_pred=y_pred_series,
            sensitive_features=sensitive,
        )
        by_group = metric_frame.by_group.to_dict(orient="index")
        return {
            f"{group}_{metric}": float(value)
            for group, metrics in by_group.items()
            for metric, value in metrics.items()
        }

    def _classification_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        scores = ModelConfig._classification_scores(self, y_true, y_pred)
        scores.update(self._compute_group_fairness_scores(y_true, y_pred))
        return scores

    def _regression_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        scores = ModelConfig._regression_scores(self, y_true, y_pred)
        scores.update(self._compute_group_fairness_scores(y_true, y_pred))
        return scores


@dataclass
class FairnessModelConfig(_FairnessBehaviorMixin, ModelConfig):
    """Fairness-aware model config for standard model workflows."""

    data: Union[FairnessDataConfig, None] = None


@dataclass
class FairnessDefenseConfig(_FairnessBehaviorMixin, DefenseConfig):
    """Fairness-aware defense config that inherits DefenseConfig."""

    data: Union[FairnessDataConfig, None] = None

    def apply_defense(self, data) -> object:
        """Apply a fairlearn or ART defense to the base estimator."""
        defense_name, _ = self._resolve_fairness_defense_spec()
        if not defense_name or not defense_name.startswith("fairlearn."):
            return super().apply_defense(data)
        return self._apply_defense(data)
