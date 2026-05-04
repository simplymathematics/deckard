import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from art.config import ART_NUMPY_DTYPE
from fairlearn.metrics import MetricFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .base import ModelConfig, logger
from .defend import DefenseConfig
from .pytorch import PytorchModelConfig
from ..data.fairness import FairlearnDataConfig
from ..utils import ConfigBase, load_class, resolve_class, resolve_torch_device

ScorerDictConfig = Any


class _FairnessBehaviorMixin:
    """Model/defense mixin for explicit fairness-aware training and scoring."""

    data: Any = None
    _model: Any = None
    scorer: Any = None
    classifier: bool = False
    probability: bool = False
    fit_params: Any = None

    def _is_torch_module_model(self) -> bool:
        model_obj = getattr(self, "_model", None)
        if model_obj is None:
            return False
        try:
            import torch

            return isinstance(model_obj, torch.nn.Module)
        except ImportError:
            return False

    def _resolve_runtime_sensitive_source(self, split: str):
        if split == "train":
            return getattr(self.data, "_sensitive_train", None)
        if split == "test":
            return getattr(self.data, "_sensitive_test", None)
        if split == "all":
            return getattr(self.data, "_sensitive_all", None)
        if split == "val":
            raise NotImplementedError(
                "Validation sensitive features are not implemented yet",
            )
        raise ValueError(f"Unsupported fairness split: {split}")

    def _resolve_scoring_split(self, mode: str) -> str:
        if mode == "train":
            return "train"
        if mode in {"test", "attack"}:
            return "test"
        if mode in {"val", "attack-val"}:
            raise NotImplementedError(
                "Validation fairness scoring is not implemented yet",
            )
        if mode == "all":
            return "all"
        raise ValueError(f"Unsupported fairness scoring mode: {mode}")

    def _validate_sensitive_series(self, sensitive, context: str):
        if sensitive is None:
            return None
        sensitive_series = pd.Series(sensitive)
        if len(sensitive_series) == 0:
            raise ValueError(f"Sensitive features are empty during {context}")
        if sensitive_series.dropna().empty:
            raise ValueError(
                f"Sensitive features are all null during {context}",
            )
        if sensitive_series.astype(str).str.strip().eq("").all():
            raise ValueError(f"Sensitive features are blank during {context}")
        return sensitive_series

    def _infer_split_from_batch(self, batch):
        if self.data is None:
            return None
        split_sources = [
            ("train", getattr(self.data, "X_train", None)),
            ("test", getattr(self.data, "X_test", None)),
            ("all", getattr(self.data, "_X", None)),
        ]
        batch_index = getattr(batch, "index", None)
        for split_name, split_data in split_sources:
            if split_data is None:
                continue
            if batch is split_data:
                return split_name
            split_index = getattr(split_data, "index", None)
            if (
                batch_index is not None
                and split_index is not None
                and batch_index.equals(
                    split_index,
                )
            ):
                return split_name
        return None

    def _resolve_sensitive_features_for_batch(
        self,
        batch,
        split: Optional[str] = None,
    ):
        if self.data is None:
            return None

        n_rows = len(batch)
        batch_index = getattr(batch, "index", None)
        resolved_split = split or self._infer_split_from_batch(batch)
        if resolved_split is None:
            return None

        sensitive = self._resolve_runtime_sensitive_source(resolved_split)
        sensitive_series = self._validate_sensitive_series(sensitive, "runtime")
        if sensitive_series is None or len(sensitive_series) != n_rows:
            return None
        if batch_index is not None:
            try:
                aligned = sensitive_series.reindex(batch_index)
                if len(aligned) == n_rows and aligned.notna().all():
                    return aligned.reset_index(drop=True)
            except Exception:
                return None
        return sensitive_series.reset_index(drop=True)

    def _method_accepts_sensitive_features(self, method) -> bool:
        try:
            params = inspect.signature(method).parameters
            if "sensitive_features" in params:
                return True
            return any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            return False

    def _call_with_optional_sensitive(self, method, X, sensitive):
        if sensitive is not None and self._method_accepts_sensitive_features(
            method,
        ):
            return method(X, sensitive_features=sensitive)
        return method(X)

    def _fit_defended_estimator(self, defended_estimator, data):
        if data is None or not hasattr(defended_estimator, "fit"):
            return defended_estimator

        if getattr(self, "data", None) is None:
            self.data = data

        sensitive = self._resolve_sensitive_features_for_batch(
            data.y_train,
            split="train",
        )
        fit_method = defended_estimator.fit
        if sensitive is not None and self._method_accepts_sensitive_features(
            fit_method,
        ):
            sensitive_arg = (
                sensitive.to_numpy() if hasattr(sensitive, "to_numpy") else sensitive
            )
            fit_method(
                data.X_train,
                data.y_train,
                sensitive_features=sensitive_arg,
            )
        else:
            fit_method(data.X_train, data.y_train)
        return defended_estimator

    def _resolve_torch_device(self, requested_device):
        try:
            return resolve_torch_device(requested_device)
        except Exception:
            return None

    def _move_torch_model_to_device(self, model_obj, requested_device):
        if requested_device is None:
            return model_obj
        try:
            import torch
        except ImportError:
            return model_obj

        if not isinstance(model_obj, torch.nn.Module):
            return model_obj

        try:
            device = self._resolve_torch_device(requested_device)
            if device is not None:
                model_obj = model_obj.to(device)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to move fairlearn model to device '%s': %s",
                requested_device,
                exc,
            )
        return model_obj

    def _resolve_fairlearn_model_param(self, spec, fallback=None):
        if spec is None:
            return fallback
        if hasattr(spec, "get_model") and callable(spec.get_model):
            try:
                return spec.get_model()
            except Exception:
                pass
        if hasattr(spec, "_model") and getattr(spec, "_model", None) is not None:
            return getattr(spec, "_model")
        if isinstance(spec, dict):
            if "model_type" in spec:
                model_type = spec.get("model_type")
                model_params = spec.get("model_params", {}) or {}
                if isinstance(model_type, str):
                    model_obj = load_class(model_type, **model_params)
                    model_obj = self._move_torch_model_to_device(
                        model_obj,
                        spec.get("device", None),
                    )
                    return model_obj
            if "name" in spec:
                spec = {
                    "_target_": spec["name"],
                    **{k: v for k, v in spec.items() if k != "name"},
                }
            if "_target_" in spec:
                target = spec.get("_target_")
                kwargs = {k: v for k, v in spec.items() if k != "_target_"}
                obj = load_class(target, **kwargs)
                if hasattr(obj, "get_model") and callable(obj.get_model):
                    try:
                        return obj.get_model()
                    except Exception:
                        return obj
                return obj
        if isinstance(spec, ConfigBase):
            if hasattr(spec, "get_model") and callable(spec.get_model):
                return spec.get_model()
            spec_dict = spec.to_dict()
            target = spec_dict.get("model_type")
            params = spec_dict.get("model_params", {})
            if isinstance(target, str):
                return load_class(target, **params)
        if isinstance(spec, str):
            if "." in spec or ":" in spec:
                return load_class(spec)
            return spec
        return spec

    def _adapt_binary_torch_predictor(self, predictor_model, data):
        """Fairlearn binary classification expects a single-score predictor output."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return predictor_model

        if not hasattr(predictor_model, "forward"):
            return predictor_model

        y_train = getattr(data, "y_train", None)
        if y_train is None:
            return predictor_model

        if isinstance(y_train, torch.Tensor):
            y_values = y_train.detach().cpu().numpy()
        else:
            y_values = np.asarray(y_train)
        if np.unique(y_values).size != 2:
            return predictor_model

        def _needs_wrap(model) -> bool:
            num_classes = getattr(model, "num_classes", None)
            if num_classes == 2:
                return True
            x_train = getattr(data, "X_train", None)
            if not isinstance(x_train, torch.Tensor) or len(x_train) == 0:
                return False
            try:
                with torch.no_grad():
                    sample = x_train[:1]
                    device = next(model.parameters()).device
                    out = model(sample.to(device))
                return bool(getattr(out, "ndim", 0) == 2 and out.shape[1] == 2)
            except Exception:
                return False

        if not _needs_wrap(predictor_model):
            return predictor_model

        class _BinaryLogitAdapter(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                out = self.base_model(x)
                if out.ndim == 1:
                    return out.reshape(-1, 1)
                if out.ndim == 2 and out.shape[1] == 1:
                    return out
                if out.ndim == 2 and out.shape[1] >= 2:
                    return out[:, 1:2]
                raise ValueError(
                    f"Unsupported predictor output shape for fairness: {out.shape}",
                )

        return _BinaryLogitAdapter(predictor_model)

    def _resolve_fairness_defense_spec(self):
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

    def _apply_fairlearn_defense(self, data):
        defense_name, defense_params = self._resolve_fairness_defense_spec()
        if not defense_name or not defense_name.startswith("fairlearn."):
            raise ValueError(
                "Fairlearn defense helper requires a fairlearn defense_name",
            )

        if getattr(self, "data", None) is None:
            self.data = data

        if self._model is None:
            raise ValueError(
                "Fairness model must have a fitted estimator before applying defense",
            )

        module_name, class_name = defense_name.rsplit(".", 1)
        try:
            defense_class = resolve_class(defense_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Could not import defense class {class_name} from {module_name}",
            ) from exc

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
            predictor_model = self._resolve_fairlearn_model_param(
                defense_params.pop("predictor_model", None),
                fallback=base_estimator,
            )
            predictor_model = self._adapt_binary_torch_predictor(
                predictor_model,
                data,
            )
            adversary_model = self._resolve_fairlearn_model_param(
                defense_params.pop("adversary_model", None),
                fallback=base_estimator,
            )
            defense_params["predictor_model"] = predictor_model
            defense_params["adversary_model"] = adversary_model
            defended_estimator = defense_class(**defense_params)
        else:
            raise NotImplementedError(
                f"Fairlearn submodule '{fairlearn_submodule}' is not supported. "
                "Expected one of: reductions, postprocessing, adversarial.",
            )

        self._apply_fit = True
        if self._apply_fit:
            defended_estimator = self._fit_defended_estimator(
                defended_estimator,
                data,
            )
        self.defense_application_time = time.process_time() - start
        return defended_estimator

    def _train(self, X: pd.DataFrame, y: pd.Series):
        if self._model is None:
            raise ValueError("Model not initialized")
        start_time = time.process_time()
        assert hasattr(self._model, "fit"), "Model does not have a fit method"
        fit_params = getattr(self, "fit_params", None) or {}
        sensitive = self._resolve_sensitive_features_for_batch(y, split="train")
        if (
            sensitive is not None
            and self._method_accepts_sensitive_features(self._model.fit)
            and "sensitive_features" not in fit_params
        ):
            fit_params = {**fit_params, "sensitive_features": sensitive}
        self._model.fit(X, y, **fit_params)
        self.training_time = time.process_time() - start_time
        self.training_n = len(y)
        logger.info(f"Model trained in {self.training_time:.2f} seconds")

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise ValueError("Model not initialized")
        sensitive = self._resolve_sensitive_features_for_batch(X, split="test")
        try:
            return self._call_with_optional_sensitive(
                self._model.predict,
                X,
                sensitive,
            )
        except TypeError as exc:
            if "loop of ufunc does not support argument" in str(
                exc,
            ) or "can't convert" in str(exc):
                X_array = np.array(X, dtype=ART_NUMPY_DTYPE)
                return self._call_with_optional_sensitive(
                    self._model.predict,
                    X_array,
                    sensitive,
                )
            raise

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise ValueError("Model not initialized")
        if not self.probability:
            raise ValueError("Model does not support probability predictions")
        sensitive = self._resolve_sensitive_features_for_batch(X, split="test")
        return self._call_with_optional_sensitive(
            self._model.predict_proba,
            X,
            sensitive,
        )

    def _resolve_sensitive_features(
        self,
        y_true: pd.Series,
        mode: str = "test",
    ):
        if self.data is None:
            return None

        y_true_series = pd.Series(y_true)
        y_true_n = len(y_true_series)
        y_index = getattr(y_true, "index", None)
        scoring_split = self._resolve_scoring_split(mode)
        sensitive = self._resolve_runtime_sensitive_source(scoring_split)
        sensitive_series = self._validate_sensitive_series(
            sensitive,
            "fairness scoring",
        )
        if sensitive_series is None or len(sensitive_series) != y_true_n:
            return None
        if y_index is not None:
            try:
                aligned = sensitive_series.reindex(y_index)
                if len(aligned) == y_true_n and aligned.notna().all():
                    return aligned.reset_index(drop=True)
            except Exception:
                return None
        return sensitive_series.reset_index(drop=True)

    def _compute_sensitive_fairness_scores(
        self,
        y_true,
        y_pred,
        mode: str = "test",
    ) -> dict:
        sensitive = self._resolve_sensitive_features(y_true, mode=mode)
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
                    labels = np.where(
                        binary_scores >= threshold,
                        high_label,
                        low_label,
                    )
                else:
                    labels = (binary_scores >= threshold).astype(int)
                return pd.Series(labels).reset_index(drop=True)

            sorted_classes = classes[np.argsort(classes)]
            pred_indices = np.argmax(y_pred_arr, axis=1)
            return pd.Series(sorted_classes[pred_indices]).reset_index(
                drop=True,
            )

        y_pred_series = _normalize_pred_labels(y_pred, y_true_series)
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
            by_group = pd.DataFrame(metric_frame.by_group).to_dict(
                orient="index",
            )
            sensitive_scores = {
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
            sensitive_accuracies = []
            for _, sensitive_df in score_df.groupby("sensitive", sort=False):
                sensitive_accuracies.append(
                    float(
                        (sensitive_df["y_true"] == sensitive_df["y_pred"]).mean(),
                    ),
                )
            if sensitive_accuracies:
                acc_arr = np.asarray(sensitive_accuracies, dtype=float)
                acc_min = float(np.min(acc_arr))
                acc_max = float(np.max(acc_arr))
                sensitive_scores["sensitive_feature_accuracy_difference"] = (
                    acc_max - acc_min
                )
                sensitive_scores["sensitive_feature_accuracy_std"] = float(
                    np.std(acc_arr),
                )
                sensitive_scores["sensitive_feature_accuracy_ratio"] = (
                    acc_min / acc_max
                    if not np.isclose(acc_max, 0.0)
                    else (1.0 if np.isclose(acc_min, 0.0) else 0.0)
                )
            return sensitive_scores

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
        by_group = pd.DataFrame(metric_frame.by_group).to_dict(orient="index")
        return {
            f"{group}_{metric}": float(value)
            for group, metrics in by_group.items()
            for metric, value in metrics.items()
        }

    def _compute_group_fairness_scores(
        self,
        y_true,
        y_pred,
        mode: str = "test",
    ) -> dict:
        return self._compute_sensitive_fairness_scores(
            y_true,
            y_pred,
            mode=mode,
        )

    def _classification_scores(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> dict:
        scores = super()._classification_scores(y_true, y_pred)
        scores.update(
            self._compute_sensitive_fairness_scores(
                y_true,
                y_pred,
                mode="test",
            ),
        )
        return scores

    def _regression_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        scores = super()._regression_scores(y_true, y_pred)
        scores.update(
            self._compute_sensitive_fairness_scores(
                y_true,
                y_pred,
                mode="test",
            ),
        )
        return scores


@dataclass(eq=False)
class FairlearnModelConfig(_FairnessBehaviorMixin, ModelConfig):
    """Fairness-aware model config for sklearn models.

    Inherits sklearn training/prediction from ModelConfig and adds
    fairness-aware scoring and defense support via _FairnessBehaviorMixin.
    """

    data: Union[FairlearnDataConfig, None] = None
    fit_params: dict = field(default_factory=dict)


@dataclass(eq=False)
class FairlearnPytorchModelConfig(_FairnessBehaviorMixin, PytorchModelConfig):
    """Fairness-aware model config for PyTorch models.

    Inherits all torch training/prediction/defense from PytorchModelConfig
    and adds fairness-aware scoring via _FairnessBehaviorMixin.
    """

    data: Union[FairlearnDataConfig, None] = None

    def _train(self, X, y):
        return PytorchModelConfig._train(self, X, y)

    def _predict(self, X):
        return PytorchModelConfig._predict(self, X)


@dataclass(eq=False)
class FairlearnDefenseConfig(_FairnessBehaviorMixin, DefenseConfig):
    """Fairness-aware defense config that inherits DefenseConfig."""

    data: Union[FairlearnDataConfig, None] = None

    def apply_defense(self, data) -> BaseEstimator:
        defense_name, _ = self._resolve_fairness_defense_spec()
        if not defense_name or not defense_name.startswith("fairlearn."):
            return super().apply_defense(data)
        return self._apply_fairlearn_defense(data)
