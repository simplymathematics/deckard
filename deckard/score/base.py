"""Core scoring primitives and default scorer profiles."""

from dataclasses import dataclass, field
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Union

import numpy as np
from omegaconf import OmegaConf

from ..utils import (
    ConfigBase,
    resolve_class,
    safe_store,
    merge_list_of_dicts,
    load_class,
)
from .pytorch import to_numpy_if_torch

logger = logging.getLogger(__name__)


@dataclass
class ScorerConfig:
    """Atomic scorer configuration."""

    score_name: str
    score_function: Any
    score_params: Dict[str, Any] = field(default_factory=dict)
    greater_is_better: bool = True
    needs_proba: bool = False

    def __post_init__(self):
        if OmegaConf.is_config(self.score_function):
            self.score_function = OmegaConf.to_container(
                self.score_function,
                resolve=True,
            )
        if isinstance(self.score_function, dict):
            score_fn_spec = dict(self.score_function)
            target = score_fn_spec.pop(
                "_target_",
                score_fn_spec.pop("name", None),
            )
            if target is None:
                raise ValueError(
                    f"Scorer '{self.score_name}' dict score_function must include '_target_' or 'name'",
                )
            args = score_fn_spec.pop("_args_", [])
            if not isinstance(args, (list, tuple)):
                args = [args]
            self.score_function = load_class(
                target,
                *list(args),
                **score_fn_spec,
            )
        if isinstance(self.score_function, str):
            self.score_function = resolve_class(self.score_function)
        if not callable(self.score_function):
            raise TypeError(
                "score_function must be callable or import path string",
            )
        if self.score_params is None:
            self.score_params = {}

    def _validate_probability_input(self, y_true, y_pred):
        """Validate probability-like inputs before metric execution."""
        y_true_arr = np.asarray(to_numpy_if_torch(y_true))
        y_pred_arr = np.asarray(to_numpy_if_torch(y_pred))

        if y_pred_arr.ndim not in (1, 2):
            raise ValueError(
                f"Probability scorer '{self.score_name}' requires 1D/2D probability input; got shape {y_pred_arr.shape}",
            )
        if y_pred_arr.shape[0] != y_true_arr.shape[0]:
            raise ValueError(
                f"Probability scorer '{self.score_name}' requires matching sample counts; got {y_pred_arr.shape[0]} predictions for {y_true_arr.shape[0]} labels",
            )
        if not np.issubdtype(y_pred_arr.dtype, np.number):
            raise ValueError(
                f"Probability scorer '{self.score_name}' requires numeric probabilities",
            )
        if np.nanmin(y_pred_arr) < -1e-12 or np.nanmax(y_pred_arr) > 1.0 + 1e-12:
            raise ValueError(
                f"Probability scorer '{self.score_name}' requires values in [0, 1]",
            )

    def _normalize_predictions_for_metric(self, y_true, y_pred):
        """Convert score/probability matrices to class labels for label-only metrics."""
        metric_name = getattr(self.score_function, "__name__", "")
        label_metrics = {
            "accuracy_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "balanced_accuracy_score",
            "jaccard_score",
            "matthews_corrcoef",
            "cohen_kappa_score",
        }

        if self.needs_proba:
            self._validate_probability_input(y_true=y_true, y_pred=y_pred)
            y_pred_arr = np.asarray(to_numpy_if_torch(y_pred))
            if y_pred_arr.ndim == 2 and metric_name == "roc_auc_score":
                if y_pred_arr.shape[1] == 1:
                    return y_pred_arr.reshape(-1)
                if y_pred_arr.shape[1] == 2:
                    return y_pred_arr[:, 1]
            return y_pred
        if metric_name not in label_metrics:
            return y_pred
        y_true_arr = np.asarray(to_numpy_if_torch(y_true))
        y_pred_arr = np.asarray(to_numpy_if_torch(y_pred))
        if y_true_arr.ndim != 1 or y_pred_arr.ndim != 2:
            return y_pred
        if not np.issubdtype(y_pred_arr.dtype, np.number):
            return y_pred

        if y_pred_arr.shape[1] == 1:
            binary_scores = y_pred_arr.reshape(-1)
            threshold = 0.5
            if np.nanmin(binary_scores) < 0.0 or np.nanmax(binary_scores) > 1.0:
                threshold = 0.0
            return (binary_scores >= threshold).astype(int)

        return np.argmax(y_pred_arr, axis=1)

    def __call__(self, y_true, y_pred, swap: bool = False, **kwargs):
        if swap:
            y_true, y_pred = y_pred, y_true
        y_true = to_numpy_if_torch(y_true)
        y_pred = to_numpy_if_torch(y_pred)
        y_pred = self._normalize_predictions_for_metric(
            y_true=y_true,
            y_pred=y_pred,
        )
        params = {**self.score_params, **kwargs}
        signature = inspect.signature(self.score_function)
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            accepted = {
                name
                for name, param in signature.parameters.items()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            accepted.discard("y_true")
            accepted.discard("y_pred")
            params = {k: v for k, v in params.items() if k in accepted}
        return self.score_function(y_true, y_pred, **params)


@dataclass(eq=False)
class ScorerDictConfig(ConfigBase):
    """Container of named ScorerConfig instances."""

    scorers: Dict[str, Union[ScorerConfig, Dict[str, Any]]] = field(
        default_factory=dict,
    )

    def __post_init__(self):
        normalized = {}
        for key, value in self.scorers.items():
            if isinstance(value, ScorerConfig):
                scorer = value
            elif isinstance(value, dict):
                scorer_data = dict(value)
                scorer = ScorerConfig(
                    score_name=scorer_data.pop("score_name", key),
                    score_function=scorer_data.pop("score_function"),
                    score_params=scorer_data.pop("score_params", {}),
                    greater_is_better=scorer_data.pop(
                        "greater_is_better",
                        True,
                    ),
                    needs_proba=scorer_data.pop("needs_proba", False),
                )
            else:
                raise TypeError(
                    f"Value for key '{key}' must be ScorerConfig or dict, got {type(value)}",
                )
            normalized[key] = scorer
        self.scorers = normalized

    def __iter__(self):
        return iter(self.scorers.items())

    def __hash__(self):
        return super().__hash__()

    def __getitem__(self, key):
        return self.scorers[key]

    def get_callables(self):
        return {key: scorer for key, scorer in self.scorers.items()}

    @classmethod
    def merge(cls, items):
        """Merge a list of scorer specs into a single ScorerDictConfig.

        Each element of *items* may be a :class:`ScorerDictConfig`, a dict
        with a ``scorers`` key, or a bare scorers dict (name → scorer spec).
        Later entries win on duplicate scorer names.
        """
        merged_scorers: dict = {}
        for item in items:
            if isinstance(item, ScorerDictConfig):
                # Already normalised – grab the inner scorer dict directly
                merged_scorers.update(item.scorers)
            else:
                # Delegate OmegaConf/dict normalisation to the shared utility
                plain = merge_list_of_dicts([item])
                if "scorers" in plain:
                    merged_scorers.update(plain["scorers"])
                else:
                    merged_scorers.update(plain)
        return cls(scorers=merged_scorers)

    @staticmethod
    def _resolve_mode_features(mode, data):
        if data is None:
            return None
        if mode == "train":
            return getattr(data, "X_train", None)
        if mode == "test":
            return getattr(data, "X_test", None)
        if mode in {"val", "attack-val"}:
            return getattr(data, "X_val", None)
        return None

    @staticmethod
    def _predict_proba_from_model(model, X):
        if model is None or X is None:
            return None

        estimator = None
        if hasattr(model, "get_model") and callable(model.get_model):
            try:
                estimator = model.get_model()
            except Exception:
                estimator = None
        if estimator is None:
            estimator = getattr(model, "_model", None)
        if estimator is None or not hasattr(estimator, "predict_proba"):
            raise ValueError(
                "Probability-required scorer configured but model does not expose predict_proba",
            )

        try:
            return estimator.predict_proba(X)
        except TypeError:
            return estimator.predict_proba(np.asarray(X, dtype=float))

    def __call__(
        self,
        mode: Literal[
            "test",
            "train",
            "attack",
            "val",
            "attack-val",
            None,
        ] = "test",
        data=None,
        model=None,
        attack=None,
        y_pred=None,
        y_true=None,
        score_file=None,
        **kwargs,
    ) -> Dict[str, Any]:
        results = {}
        if score_file is not None and Path(score_file).exists():
            results = self.load_scores(score_file)

        if y_pred is not None:
            if y_true is None:
                raise AssertionError(
                    "If y_pred is provided, y_true must also be provided.",
                )
        else:
            if mode == "test":
                y_true = data.y_test
                y_pred = getattr(model, "test_predictions", None)
                if y_pred is None:
                    y_pred = getattr(model, "predictions", None)
            elif mode == "train":
                y_true = data.y_train
                y_pred = model.training_predictions
            elif mode == "attack":
                y_true = data.y_test[: attack.attack_size]
                y_pred = attack.attack_predictions
            elif mode == "val":
                y_true = data.y_val
                y_pred = model.val_predictions
            elif mode == "attack-val":
                y_true = data.y_val
                y_pred = attack.attack_predictions
            elif y_true is None:
                raise AssertionError("y_true must be provided if mode is None")

        if attack is not None:
            for key, value in kwargs.items():
                if value == "{attack}":
                    kwargs[key] = attack._attack

        y_proba = kwargs.pop("y_proba", None)

        runtime_kwargs = {
            **kwargs,
            "data": data,
            "model": model,
            "attack": attack,
            "mode": mode,
        }

        for key, scorer in self.scorers.items():
            scored_key = key
            if mode == "train":
                scored_key = f"training_{key}"
            elif mode == "attack":
                scored_key = f"attack_{key}"
            if results.get(scored_key) is None:
                metric_input = y_pred
                if scorer.needs_proba:
                    if y_proba is not None:
                        metric_input = y_proba
                    else:
                        X_mode = self._resolve_mode_features(
                            mode=mode,
                            data=data,
                        )
                        if X_mode is not None and model is not None:
                            metric_input = self._predict_proba_from_model(
                                model=model,
                                X=X_mode,
                            )
                        else:
                            raise ValueError(
                                f"Scorer '{key}' requires probabilities from predict_proba; provide y_proba or pass model+data context",
                            )
                results[scored_key] = scorer(
                    y_true=y_true,
                    y_pred=metric_input,
                    **runtime_kwargs,
                )

        if score_file is not None:
            self.save_scores(results, score_file)
        return results


def build_scorer(cfg: ScorerConfig):
    return cfg if isinstance(cfg, ScorerConfig) else ScorerConfig(**cfg)


def build_scorer_dict(cfg: ScorerDictConfig):
    return cfg if isinstance(cfg, ScorerDictConfig) else ScorerDictConfig(**cfg)


@dataclass(eq=False)
class DefaultClassifierConfig(ScorerDictConfig):
    scorers: Dict[str, ScorerConfig] = field(
        default_factory=lambda: {
            "accuracy": ScorerConfig(
                score_name="accuracy",
                score_function="sklearn.metrics.accuracy_score",
            ),
            "precision": ScorerConfig(
                score_name="precision",
                score_function="sklearn.metrics.precision_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "recall": ScorerConfig(
                score_name="recall",
                score_function="sklearn.metrics.recall_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "f1": ScorerConfig(
                score_name="f1",
                score_function="sklearn.metrics.f1_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "roc_auc": ScorerConfig(
                score_name="roc_auc",
                score_function="sklearn.metrics.roc_auc_score",
                score_params={"average": "weighted", "multi_class": "ovr"},
                needs_proba=True,
            ),
            "log_loss": ScorerConfig(
                score_name="log_loss",
                score_function="sklearn.metrics.log_loss",
                needs_proba=True,
            ),
        },
    )


@dataclass(eq=False)
class DefaultRegressorConfig(ScorerDictConfig):
    scorers: Dict[str, ScorerConfig] = field(
        default_factory=lambda: {
            "mse": ScorerConfig(
                score_name="mse",
                score_function="sklearn.metrics.mean_squared_error",
                greater_is_better=False,
            ),
            "mae": ScorerConfig(
                score_name="mae",
                score_function="sklearn.metrics.mean_absolute_error",
                greater_is_better=False,
            ),
            "r2": ScorerConfig(
                score_name="r2",
                score_function="sklearn.metrics.r2_score",
            ),
        },
    )


@dataclass(eq=False)
class DefaultPytorchClassifierConfig(ScorerDictConfig):
    """Default classifier scorers for PyTorch models.

    PyTorch model wrappers often expose logits but not ``predict_proba``. This
    default avoids probability-required metrics so automatic scoring works out
    of the box.
    """

    scorers: Dict[str, ScorerConfig] = field(
        default_factory=lambda: {
            "accuracy": ScorerConfig(
                score_name="accuracy",
                score_function="sklearn.metrics.accuracy_score",
            ),
            "precision": ScorerConfig(
                score_name="precision",
                score_function="sklearn.metrics.precision_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "recall": ScorerConfig(
                score_name="recall",
                score_function="sklearn.metrics.recall_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "f1": ScorerConfig(
                score_name="f1",
                score_function="sklearn.metrics.f1_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
        },
    )


@dataclass(eq=False)
class DefaultPytorchRegressorConfig(ScorerDictConfig):
    """Default regressor scorers for PyTorch models."""

    scorers: Dict[str, ScorerConfig] = field(
        default_factory=lambda: {
            "mse": ScorerConfig(
                score_name="mse",
                score_function="sklearn.metrics.mean_squared_error",
                greater_is_better=False,
            ),
            "mae": ScorerConfig(
                score_name="mae",
                score_function="sklearn.metrics.mean_absolute_error",
                greater_is_better=False,
            ),
            "r2": ScorerConfig(
                score_name="r2",
                score_function="sklearn.metrics.r2_score",
            ),
        },
    )


safe_store(group="score", name="classification", node=DefaultClassifierConfig)
safe_store(group="score", name="regression", node=DefaultRegressorConfig)
safe_store(
    group="score",
    name="pytorch_classification",
    node=DefaultPytorchClassifierConfig,
)
safe_store(
    group="score",
    name="pytorch_regression",
    node=DefaultPytorchRegressorConfig,
)


__all__ = [
    "safe_store",
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultClassifierConfig",
    "DefaultRegressorConfig",
    "DefaultPytorchClassifierConfig",
    "DefaultPytorchRegressorConfig",
    "build_scorer",
    "build_scorer_dict",
]
