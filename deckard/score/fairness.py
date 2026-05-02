"""Fairness-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Union, cast

import numpy as np
import pandas as pd

try:
    from fairlearn.metrics import MetricFrame
except ImportError:  # pragma: no cover
    MetricFrame = None

from .base import ScorerConfig, ScorerDictConfig, safe_store

__all__ = [
    "fairness_demographic_parity_difference",
    "fairness_equalized_odds_difference",
    "fairness_group_mean_prediction_difference",
    "fairness_group_mae_difference",
    "fairness_group_mse_difference",
    "FairnessScoreDictConfig",
    "DefaultFairnessClassificationConfig",
    "DefaultFairnessRegressionConfig",
    "DefaultFairnessConfig",
]


def _resolve_sensitive_features(data, y_true, mode="test"):
    if data is None:
        return None
    if mode == "train":
        sensitive = getattr(data, "_sensitive_train", None)
    elif mode in {"test", "attack"}:
        sensitive = getattr(data, "_sensitive_test", None)
    elif mode in {"val", "attack-val"}:
        raise NotImplementedError(
            "Validation sensitive features are not implemented yet",
        )
    elif mode == "all":
        sensitive = getattr(data, "_sensitive_all", None)
    else:
        raise ValueError(f"Unsupported fairness scoring mode: {mode}")

    if sensitive is None or len(sensitive) != len(y_true):
        return None
    return sensitive


def fairness_demographic_parity_difference(y_true, y_pred, data=None, **kwargs):
    """Compute demographic parity difference for fairness-aware configurations."""
    try:
        from fairlearn.metrics import demographic_parity_difference
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Fairness scorer requires optional dependency deckard[fairlearn]",
        ) from exc

    sensitive_features = kwargs.get("sensitive_features")
    if sensitive_features is None:
        sensitive_features = _resolve_sensitive_features(
            data,
            y_true,
            mode=kwargs.get("mode", "test"),
        )
    if sensitive_features is None:
        raise ValueError("sensitive_features are required for fairness scoring")

    return float(
        demographic_parity_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        ),
    )


def fairness_equalized_odds_difference(y_true, y_pred, data=None, **kwargs):
    """Compute equalized odds difference for fairness-aware configurations."""
    try:
        from fairlearn.metrics import equalized_odds_difference
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Fairness scorer requires optional dependency deckard[fairlearn]",
        ) from exc

    sensitive_features = kwargs.get("sensitive_features")
    if sensitive_features is None:
        sensitive_features = _resolve_sensitive_features(
            data,
            y_true,
            mode=kwargs.get("mode", "test"),
        )
    if sensitive_features is None:
        raise ValueError("sensitive_features are required for fairness scoring")

    return float(
        equalized_odds_difference(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        ),
    )


def _resolve_sensitive_from_kwargs_or_data(y_true, data=None, **kwargs):
    sensitive_features = kwargs.get("sensitive_features")
    if sensitive_features is None:
        sensitive_features = _resolve_sensitive_features(
            data,
            y_true,
            mode=kwargs.get("mode", "test"),
        )
    if sensitive_features is None:
        raise ValueError("sensitive_features are required for fairness scoring")
    return np.asarray(sensitive_features)


def _flatten_metric_frame_by_group(by_group: pd.DataFrame) -> dict:
    """Flatten MetricFrame.by_group into {group_metric: value} keys."""
    rows = by_group.to_dict(orient="index")
    flattened = {}
    for group, metrics in rows.items():
        group_label = str(group)
        for metric_name, value in metrics.items():
            flattened[f"{group_label}_{metric_name}"] = float(value)
    return flattened


def _series_like_to_float_dict(values) -> dict:
    if isinstance(values, pd.Series):
        return {str(key): float(value) for key, value in values.items()}
    return {"value": float(values)}


@dataclass
class FairnessScoreDictConfig(ScorerDictConfig):
    """ScorerDictConfig variant that computes fairness metrics through MetricFrame.

    Use ``group_scorers`` to provide configurable metric callables evaluated per
    sensitive group. Standard ``scorers`` are still supported and are evaluated
    first via ``ScorerDictConfig``.
    """

    group_scorers: Dict[
        str,
        Union[ScorerConfig, Dict[str, Any], str, Callable],
    ] = field(default_factory=dict)
    group_reduction: Literal["difference", "ratio", "none"] = "difference"
    group_reduction_method: Literal["between_groups", "to_overall"] = "between_groups"
    include_group_overall: bool = False
    include_group_by_group: bool = True

    def __post_init__(self):
        super().__post_init__()
        normalized = {}
        for key, value in self.group_scorers.items():
            if isinstance(value, ScorerConfig):
                scorer = value
            elif isinstance(value, dict):
                scorer_data = dict(value)
                scorer = ScorerConfig(
                    score_name=scorer_data.pop("score_name", key),
                    score_function=scorer_data.pop("score_function"),
                    score_params=scorer_data.pop("score_params", {}),
                    greater_is_better=scorer_data.pop("greater_is_better", True),
                    needs_proba=scorer_data.pop("needs_proba", False),
                )
            elif isinstance(value, str) or callable(value):
                scorer = ScorerConfig(score_name=key, score_function=value)
            else:
                raise TypeError(
                    f"Value for key '{key}' must be ScorerConfig, dict, str, or callable. Got {type(value)}",
                )
            normalized[key] = scorer
        self.group_scorers = normalized

    def _build_metric_frame(self, y_true, y_pred, sensitive_features):
        if MetricFrame is None:
            raise ImportError(
                "Fairness scorer requires optional dependency deckard[fairlearn]",
            )
        metrics = {
            key: (lambda yt, yp, scorer=scorer: scorer(y_true=yt, y_pred=yp))
            for key, scorer in cast(Dict[str, ScorerConfig], self.group_scorers).items()
        }
        return MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

    def __call__(
        self,
        mode: Literal["test", "train", "attack", "val", "attack-val", None] = "test",
        data=None,
        model=None,
        attack=None,
        y_pred=None,
        y_true=None,
        score_file=None,
        **kwargs,
    ) -> Dict[str, Any]:
        results = super().__call__(
            mode=mode,
            data=data,
            model=model,
            attack=attack,
            y_pred=y_pred,
            y_true=y_true,
            score_file=score_file,
            **kwargs,
        )
        if not self.group_scorers:
            return results

        resolved_mode = "test" if mode is None else mode
        sensitive_features = kwargs.get("sensitive_features")
        if sensitive_features is None:
            sensitive_features = _resolve_sensitive_features(
                data,
                y_true,
                mode=resolved_mode,
            )
        if sensitive_features is None:
            raise ValueError("sensitive_features are required for fairness scoring")

        metric_frame = self._build_metric_frame(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        if self.include_group_overall:
            overall = metric_frame.overall
            if isinstance(overall, pd.Series):
                for metric_name, value in overall.items():
                    results[f"{metric_name}_overall"] = float(value)
            else:
                overall_series = _series_like_to_float_dict(cast(Any, overall))
                if len(overall_series) == 1 and "value" in overall_series:
                    overall_value = overall_series["value"]
                    for metric_name in self.group_scorers.keys():
                        results[f"{metric_name}_overall"] = overall_value
                else:
                    for metric_name, value in overall_series.items():
                        results[f"{metric_name}_overall"] = value

        if self.include_group_by_group:
            results.update(_flatten_metric_frame_by_group(pd.DataFrame(metric_frame.by_group)))

        if self.group_reduction == "difference":
            reduced = metric_frame.difference(method=self.group_reduction_method)
            for metric_name, value in _series_like_to_float_dict(reduced).items():
                results[f"{metric_name}_difference"] = value
        elif self.group_reduction == "ratio":
            reduced = metric_frame.ratio(method=self.group_reduction_method)
            for metric_name, value in _series_like_to_float_dict(reduced).items():
                results[f"{metric_name}_ratio"] = value
        elif self.group_reduction != "none":
            raise ValueError(
                "group_reduction must be one of {'difference', 'ratio', 'none'}",
            )

        return results


def _group_metric_difference(y_true, y_pred, sensitive_features, metric_fn):
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    groups = np.asarray(sensitive_features)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return 0.0

    group_scores = []
    for group_value in unique_groups:
        mask = groups == group_value
        if not np.any(mask):
            continue
        group_scores.append(float(metric_fn(y_true_arr[mask], y_pred_arr[mask])))

    if len(group_scores) < 2:
        return 0.0
    return float(max(group_scores) - min(group_scores))


def fairness_group_mean_prediction_difference(y_true, y_pred, data=None, **kwargs):
    """Compute disparity in mean prediction across sensitive groups."""
    sensitive_features = _resolve_sensitive_from_kwargs_or_data(
        y_true=y_true,
        data=data,
        **kwargs,
    )
    groups = np.asarray(sensitive_features)
    y_pred_arr = np.asarray(y_pred)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return 0.0

    means = []
    for group_value in unique_groups:
        mask = groups == group_value
        if np.any(mask):
            means.append(float(np.mean(y_pred_arr[mask])))
    if len(means) < 2:
        return 0.0
    return float(max(means) - min(means))


def fairness_group_mae_difference(y_true, y_pred, data=None, **kwargs):
    """Compute disparity in MAE across sensitive groups."""
    from sklearn.metrics import mean_absolute_error

    sensitive_features = _resolve_sensitive_from_kwargs_or_data(
        y_true=y_true,
        data=data,
        **kwargs,
    )
    return _group_metric_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        metric_fn=mean_absolute_error,
    )


def fairness_group_mse_difference(y_true, y_pred, data=None, **kwargs):
    """Compute disparity in MSE across sensitive groups."""
    from sklearn.metrics import mean_squared_error

    sensitive_features = _resolve_sensitive_from_kwargs_or_data(
        y_true=y_true,
        data=data,
        **kwargs,
    )
    return _group_metric_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        metric_fn=mean_squared_error,
    )


@dataclass
class DefaultFairnessClassificationConfig(FairnessScoreDictConfig):
    """Default scorer set for classification fairness workflows."""

    scorers: Dict[str, Union[ScorerConfig, Dict[str, Any]]] = field(
        default_factory=lambda: {
            "demographic_parity_difference": ScorerConfig(
                score_name="demographic_parity_difference",
                score_function="deckard.score.fairness.fairness_demographic_parity_difference",
                greater_is_better=False,
            ),
            "equalized_odds_difference": ScorerConfig(
                score_name="equalized_odds_difference",
                score_function="deckard.score.fairness.fairness_equalized_odds_difference",
                greater_is_better=False,
            ),
        },
    )


@dataclass
class DefaultFairnessRegressionConfig(FairnessScoreDictConfig):
    """Default scorer set for regression fairness workflows."""

    scorers: Dict[str, Union[ScorerConfig, Dict[str, Any]]] = field(
        default_factory=lambda: {
            "group_mean_prediction_difference": ScorerConfig(
                score_name="group_mean_prediction_difference",
                score_function="deckard.score.fairness.fairness_group_mean_prediction_difference",
                greater_is_better=False,
            ),
            "group_mae_difference": ScorerConfig(
                score_name="group_mae_difference",
                score_function="deckard.score.fairness.fairness_group_mae_difference",
                greater_is_better=False,
            ),
            "group_mse_difference": ScorerConfig(
                score_name="group_mse_difference",
                score_function="deckard.score.fairness.fairness_group_mse_difference",
                greater_is_better=False,
            ),
        },
    )


# Backward-compatible class alias retained for Python imports.
DefaultFairnessConfig = DefaultFairnessClassificationConfig


safe_store(group="score", name="fairness-classification", node=DefaultFairnessClassificationConfig)
safe_store(group="score", name="fairness-regression", node=DefaultFairnessRegressionConfig)
