def as_group_scorer(scorer_dict, *, group_reduction="difference", group_reduction_method="between_groups", include_group_overall=True, include_group_by_group=True, **kwargs):
    """
    Wrap any ScorerDictConfig (or dict of metrics) to enable MetricFrame group scoring at runtime.
    Returns a FairlearnScoreDictConfig with group_scorers auto-populated from the main scorers.
    """
    if isinstance(scorer_dict, ScorerDictConfig):
        scorers = scorer_dict.scorers
    elif isinstance(scorer_dict, dict):
        scorers = dict(scorer_dict)
    else:
        raise TypeError("scorer_dict must be a ScorerDictConfig or dict")
    group_reduction_lit = group_reduction if group_reduction in ("difference", "ratio", "none") else "difference"
    group_reduction_method_lit = group_reduction_method if group_reduction_method in ("between_groups", "to_overall") else "between_groups"
    return FairlearnScoreDictConfig(
        scorers=scorers,
        group_scorers={},
        group_reduction=group_reduction_lit,  # type: ignore
        group_reduction_method=group_reduction_method_lit,  # type: ignore
        include_group_overall=include_group_overall,
        include_group_by_group=include_group_by_group,
        **kwargs
    )
"""Fairness-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
from collections.abc import Sized
from typing import TYPE_CHECKING, Any, Callable, Literal, Union, cast

import numpy as np
import pandas as pd
from omegaconf import ListConfig

try:
    from fairlearn.metrics import MetricFrame
except ImportError:  # pragma: no cover
    MetricFrame = None

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
    )
except ImportError:  # pragma: no cover
    demographic_parity_difference = None
    equalized_odds_difference = None

from .base import (
    ScorerConfig,
    ScorerDictConfig,
    _TaskAwareScorerMixin,
    _resolve_yt_yp,
    safe_store,
)
from ..utils import coerce_to_list, merge_list_of_dicts
from ..data import DataConfig

if TYPE_CHECKING:
    from ..attack import AttackConfig
    from ..model import ModelConfig

__all__ = [
    "_FairnessScorerMixin",
    "fairness_demographic_parity_difference",
    "fairness_equalized_odds_difference",
    "fairness_group_mean_prediction_difference",
    "fairness_group_mae_difference",
    "fairness_group_mse_difference",
    "FairlearnScoreDictConfig",
    "DefaultFairlearnScoreConfig",
    "DefaultFairlearnClassificationConfig",
    "DefaultFairlearnRegressionConfig",
    "DefaultFairlearnScoreConfig",
]

FairnessMode = Literal["test", "train", "attack", "val", "attack-val", "all"]
ControlFeaturesLike = Union[pd.Series, pd.DataFrame, np.ndarray, None]
SampleParamsLike = Union[dict[str, Any], dict[str, dict[str, Any]], None]
RandomStateLike = Union[int, np.random.RandomState, None]


def _resolve_sensitive_features(
    data: DataConfig | None,
    y_true: Any,
    mode: FairnessMode = "test",
) -> Any | None:
    if data is None:
        return None
    if mode == "train":
        sensitive = getattr(data, "_sensitive_train", None)
    elif mode in {"test", "attack"}:
        sensitive = getattr(data, "_sensitive_test", None)
    elif mode in {"val", "attack-val"}:
        sensitive = getattr(data, "_sensitive_val", None)
    elif mode == "all":
        sensitive = getattr(data, "_sensitive_all", None)
    else:
        raise ValueError(f"Unsupported fairness scoring mode: {mode}")

    if not hasattr(y_true, "__len__"):
        return None
    if sensitive is not None and not hasattr(sensitive, "__len__"):
        return None
    if sensitive is None:
        return None
    if len(cast("Sized", sensitive)) != len(cast("Sized", y_true)):
        return None
    return sensitive


def fairness_demographic_parity_difference(
    y_true: Any,
    y_pred: Any,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute demographic parity difference for fairness-aware configurations."""
    if demographic_parity_difference is None:
        raise ImportError(
            "Fairness scorer requires optional dependency deckard[fairlearn]",
        )

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


def fairness_equalized_odds_difference(
    y_true: Any,
    y_pred: Any,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute equalized odds difference for fairness-aware configurations."""
    if equalized_odds_difference is None:
        raise ImportError(
            "Fairness scorer requires optional dependency deckard[fairlearn]",
        )

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


def _resolve_sensitive_from_kwargs_or_data(
    y_true: Any,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> np.ndarray:
    sensitive_features = kwargs.get("sensitive_features")
    if sensitive_features is None:
        sensitive_features = _resolve_sensitive_features(
            data,
            y_true,
            mode=kwargs.get("mode", "test"),
        )
    if sensitive_features is None:
        raise ValueError("sensitive_features are required for fairness scoring")
    sensitive_arr = np.asarray(sensitive_features)
    if hasattr(y_true, "__len__") and len(sensitive_arr) != len(y_true):
        raise ValueError(f"Length of sensitive_features ({len(sensitive_arr)}) does not match y_true ({len(y_true)})")
    return sensitive_arr


def _flatten_metric_frame_by_group(by_group: pd.DataFrame) -> dict[str, float]:
    """Flatten MetricFrame.by_group into {group_metric: value} keys."""
    rows = by_group.to_dict(orient="index")
    flattened = {}
    for group, metrics in rows.items():
        group_label = str(group)
        for metric_name, value in metrics.items():
            flattened[f"{group_label}_{metric_name}"] = float(value)
    return flattened


def _series_like_to_float_dict(values: Any) -> dict[str, float]:
    if isinstance(values, pd.DataFrame):
        flattened = {}
        for row_key, row_values in values.to_dict(orient="index").items():
            row_label = str(row_key)
            for col_key, col_val in row_values.items():
                flattened[f"{row_label}_{col_key}"] = float(col_val)
        return flattened
    if isinstance(values, pd.Series):
        return {str(key): float(value) for key, value in values.items()}
    scalar_value = cast(Union[float, int, np.floating, np.integer], values)
    return {"value": float(scalar_value)}


class _FairnessScorerMixin:
    """Mixin that adds MetricFrame group scoring to any :class:`ScorerDictConfig` subclass.

    Override ``__call__`` to first run the base scorer (via ``super().__call__()``),
    then compute per-group metrics using MetricFrame and merge them into the
    results dict.

    The concrete class **must** declare these attributes as dataclass fields:

    * ``group_scorers`` – dict of scorer callables run inside MetricFrame.
    * ``group_reduction`` – ``"difference"`` | ``"ratio"`` | ``"none"``.
    * ``group_reduction_method`` – ``"between_groups"`` | ``"to_overall"``.
    * ``include_group_overall`` – whether to include the overall aggregate.
    * ``include_group_by_group`` – whether to include per-group values.
    * ``control_features``, ``sample_params``, ``n_boot``, ``ci_quantiles``,
      ``random_state`` – forwarded to MetricFrame as-is.

    Composition example::

        @dataclass(eq=False)
        class FairnessClassifier(_FairnessScorerMixin, DefaultClassifierConfig):
            group_scorers: dict = field(default_factory=lambda: { ... })
            group_reduction: str = "difference"
            ...
            def __post_init__(self):
                super().__post_init__()
                self._normalize_group_scorers_input()
                self._coerce_group_scorers()
    """

    def _normalize_group_scorers_input(self) -> None:
        if not isinstance(self.group_scorers, (list, ListConfig)):
            return
        merged_group_scorers: dict = {}
        for item in coerce_to_list(self.group_scorers):
            plain = merge_list_of_dicts([item])
            if "group_scorers" in plain:
                nested = plain["group_scorers"]
                if not isinstance(nested, dict):
                    raise TypeError(
                        "group_scorers wrapper must contain a dict under 'group_scorers'",
                    )
                merged_group_scorers.update(nested)
            else:
                merged_group_scorers.update(plain)
        self.group_scorers = merged_group_scorers

    def _coerce_group_scorers(self) -> None:
        normalized = {}
        # If group_scorers is empty, use all main scorers as group scorers by default
        group_source = self.group_scorers if self.group_scorers else getattr(self, 'scorers', {})
        for key, value in group_source.items():
            if isinstance(value, ScorerConfig):
                normalized[key] = value
            elif isinstance(value, ScorerDictConfig):
                for nested_key, nested_scorer in value.get_callables().items():
                    normalized[nested_key] = nested_scorer
            elif isinstance(value, dict):
                scorer_data = dict(value)
                normalized[key] = ScorerConfig(
                    score_name=scorer_data.pop("score_name", key),
                    score_function=scorer_data.pop("score_function"),
                    score_params=scorer_data.pop("score_params", {}),
                    greater_is_better=scorer_data.pop("greater_is_better", True),
                    needs_proba=scorer_data.pop("needs_proba", False),
                )
            elif isinstance(value, str) or callable(value):
                normalized[key] = ScorerConfig(score_name=key, score_function=value)
            else:
                raise TypeError(
                    f"Value for key '{key}' must be ScorerConfig, ScorerDictConfig, dict, "
                    f"str, or callable.  Got {type(value)}",
                )
        self.group_scorers = normalized

    def _build_metric_frame(
        self,
        y_true: Any,
        y_pred: Any,
        sensitive_features: Any,
        scorer_kwargs: Union[dict[str, Any], None] = None,
        control_features: ControlFeaturesLike = None,
        sample_params: SampleParamsLike = None,
        n_boot: int | None = None,
        ci_quantiles: list[float] | None = None,
        random_state: RandomStateLike = None,
    ) -> Any:
        if MetricFrame is None:
            raise ImportError(
                "Fairness scorer requires optional dependency deckard[fairlearn]",
            )
        scorer_kwargs = scorer_kwargs or {}
        scorer_kwargs_dict: dict[str, Any] = dict(scorer_kwargs)
        metrics_keys = list(cast(dict[str, ScorerConfig], self.group_scorers).keys())
        if isinstance(sample_params, dict):
            sample_param_keys = set(sample_params.keys())
            if not sample_param_keys.issubset(set(metrics_keys)):
                sample_params = {
                    metric_name: dict(sample_params) for metric_name in metrics_keys
                }
        metrics = {
            key: (
                self._make_metric_with_sensitive(scorer, scorer_kwargs_dict)
            )
            for key, scorer in cast(dict[str, ScorerConfig], self.group_scorers).items()
        }
        # Defensive: If metrics is empty, return None to avoid constructing MetricFrame with no metrics
        if not metrics:
            return None

        return MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            control_features=control_features,
            sample_params=sample_params,
            n_boot=n_boot,
            ci_quantiles=ci_quantiles,
            random_state=random_state,
        )

    def _make_metric_with_sensitive(self, scorer, scorer_kwargs_dict):
        data = getattr(self, 'data', None)
        import numpy as np
        def metric_callable_call(self, y_true, y_pred, **sample_kwargs):
            sensitive = sample_kwargs.get("sensitive_features")
            if sensitive is None:
                sensitive = _resolve_sensitive_features(self.data, y_true, mode=self.scorer_kwargs_dict.get("mode", "test"))
            try:
                # Avoid passing sensitive_features twice
                call_kwargs = {k: v for k, v in sample_kwargs.items() if k != "sensitive_features"}
                call_kwargs.update(self.scorer_kwargs_dict)
                if "sensitive_features" not in call_kwargs:
                    call_kwargs["sensitive_features"] = sensitive
                result = cast(
                    Callable[..., Any],
                    self.scorer,
                )(
                    y_true=y_true,
                    y_pred=y_pred,
                    data=self.data,
                    **call_kwargs,
                )
                if result is None:
                    print(f"[DEBUG] Metric function returned None for scorer {self.scorer} with y_true={y_true}, y_pred={y_pred}, sensitive={sensitive}")
                    return np.nan
                return result
            except Exception as e:
                print(f"[DEBUG] Exception in metric function for scorer {self.scorer}: {e}")
                return np.nan

        class MetricCallable:
            def __init__(self, scorer, data, scorer_kwargs_dict):
                self.scorer = scorer
                self.data = data
                self.scorer_kwargs_dict = scorer_kwargs_dict
            def __call__(self, y_true, y_pred, **sample_kwargs):
                return metric_callable_call(self, y_true, y_pred, **sample_kwargs)
        return MetricCallable(scorer, data, scorer_kwargs_dict)
    def __call__(
        self,
        mode: Literal["test", "train", "attack", "val", "attack-val", None] = "test",
        data: DataConfig | None = None,
        model: "ModelConfig | None" = None,
        attack: "AttackConfig | None" = None,
        y_pred: Any | None = None,
        y_true: Any | None = None,
        score_file: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Step 1: resolve y_true/y_pred for both main and group metrics.
        if data is None and (y_true is None or y_pred is None):
            raise ValueError("data must be provided when y_true/y_pred are not passed directly")
        resolved_y_true, resolved_y_pred = _resolve_yt_yp(
            mode, cast(DataConfig, data), model, attack, y_pred, y_true,
        )

        # Step 2: resolve sensitive features once.
        resolved_mode = "test" if mode is None else mode
        sensitive_features = kwargs.get("sensitive_features")
        if sensitive_features is None:
            sensitive_features = _resolve_sensitive_features(
                data,
                resolved_y_true,
                mode=resolved_mode,
            )
        if sensitive_features is None:
            raise ValueError("sensitive_features are required for fairness scoring")

        # Step 3: run base ScorerDictConfig scorers using resolved arrays.
        results = ScorerDictConfig.__call__(
            cast(ScorerDictConfig, self),
            mode=mode,
            data=data,
            model=model,
            attack=attack,
            y_pred=resolved_y_pred,
            y_true=resolved_y_true,
            score_file=score_file,
            **kwargs,
        )
        if not self.group_scorers:
            return results

        # Step 4: build MetricFrame and populate results using resolved arrays.
        self_cfg = cast("FairlearnScoreDictConfig", self)
        control_features = cast(
            ControlFeaturesLike,
            kwargs.pop("control_features", self_cfg.control_features),
        )
        sample_params = cast(
            SampleParamsLike,
            kwargs.pop("sample_params", self_cfg.sample_params),
        )
        n_boot = cast(int | None, kwargs.pop("n_boot", self_cfg.n_boot))
        ci_quantiles = cast(
            list[float] | None,
            kwargs.pop("ci_quantiles", self_cfg.ci_quantiles),
        )
        random_state = cast(
            RandomStateLike,
            kwargs.pop("random_state", self_cfg.random_state),
        )

        metric_frame = self._build_metric_frame(
            y_true=resolved_y_true,
            y_pred=resolved_y_pred,
            sensitive_features=sensitive_features,
            scorer_kwargs=kwargs,
            control_features=control_features,
            sample_params=sample_params,
            n_boot=n_boot,
            ci_quantiles=ci_quantiles,
            random_state=random_state,
        )

        if self_cfg.include_group_overall:
            overall = metric_frame.overall
            if isinstance(overall, pd.Series):
                for metric_name, value in overall.items():
                    results[f"{metric_name}_overall"] = float(value)
            else:
                overall_series = _series_like_to_float_dict(cast(Any, overall))
                if len(overall_series) == 1 and "value" in overall_series:
                    overall_value = overall_series["value"]
                    for metric_name in self_cfg.group_scorers.keys():
                        results[f"{metric_name}_overall"] = overall_value
                else:
                    for metric_name, value in overall_series.items():
                        results[f"{metric_name}_overall"] = value

        if self_cfg.include_group_by_group:
            results.update(
                _flatten_metric_frame_by_group(pd.DataFrame(metric_frame.by_group)),
            )

        # Only apply reduction to metrics that are scalar per group (not group metric functions)
        def is_scalar_metric(metric_name):
            # Heuristic: group metric functions have 'group_' prefix and '_difference' or '_ratio' suffix
            return not (metric_name.startswith("group_") and metric_name.endswith(("_difference", "_ratio")))

        if self_cfg.group_reduction == "difference":
            reduced = metric_frame.difference(method=self_cfg.group_reduction_method)
            for metric_name, value in _series_like_to_float_dict(reduced).items():
                if is_scalar_metric(metric_name):
                    results[f"{metric_name}_difference"] = value
        elif self_cfg.group_reduction == "ratio":
            reduced = metric_frame.ratio(method=self_cfg.group_reduction_method)
            for metric_name, value in _series_like_to_float_dict(reduced).items():
                if is_scalar_metric(metric_name):
                    results[f"{metric_name}_ratio"] = value
        elif self_cfg.group_reduction != "none":
            raise ValueError(
                "group_reduction must be one of {'difference', 'ratio', 'none'}",
            )

        return results


@dataclass(eq=False)
class FairlearnScoreDictConfig(_FairnessScorerMixin, ScorerDictConfig):
    """ScorerDictConfig variant that computes fairness metrics through MetricFrame.

    Composes ``_FairnessScorerMixin`` (group scoring) with ``ScorerDictConfig``
    (standard scorer evaluation).  Use ``group_scorers`` to provide configurable
    metric callables evaluated per sensitive group via MetricFrame.  Standard
    ``scorers`` are still evaluated first.
    """

    group_scorers: dict[
        str,
        Union[
            ScorerConfig,
            ScorerDictConfig,
            dict[str, Any],
            str,
            Callable[..., Any],
        ],
    ] = field(default_factory=dict)
    group_reduction: Literal["difference", "ratio", "none"] = "difference"
    group_reduction_method: Literal["between_groups", "to_overall"] = "between_groups"
    include_group_overall: bool = False
    include_group_by_group: bool = True
    control_features: ControlFeaturesLike = None
    sample_params: SampleParamsLike = None
    n_boot: int | None = None
    ci_quantiles: list[float] | None = None
    random_state: RandomStateLike = None

    def __post_init__(self):
        super().__post_init__()
        self._normalize_group_scorers_input()
        self._coerce_group_scorers()
        if not self.group_scorers:
            raise ValueError(
                "group_scorers must not be empty. Either provide group_scorers explicitly or ensure scorers is non-empty. "
                "This is required for MetricFrame group scoring."
            )


def _group_metric_difference(
    y_true: Any,
    y_pred: Any,
    sensitive_features: Any,
    metric_fn: Callable[..., Any],
) -> float:
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
        metric_value = cast(
            Union[float, int, np.floating, np.integer],
            metric_fn(y_true_arr[mask], y_pred_arr[mask]),
        )
        group_scores.append(
            float(metric_value),
        )

    if len(group_scores) < 2:
        return 0.0
    return float(max(group_scores) - min(group_scores))


def fairness_group_mean_prediction_difference(
    y_true: Any,
    y_pred: Any,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
    """Compute disparity in mean prediction across sensitive groups."""
    sensitive_features = _resolve_sensitive_from_kwargs_or_data(
        y_true=y_true,
        data=data,
        **kwargs,
    )
    groups = np.asarray(sensitive_features)
    y_pred_arr = np.asarray(y_pred)
    print(f"[DEBUG] fairness_group_mean_prediction_difference: y_true.shape={np.shape(y_true)}, y_pred.shape={np.shape(y_pred_arr)}, sensitive_features.shape={np.shape(groups)}")
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


def fairness_group_mae_difference(
    y_true: Any,
    y_pred: Any,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
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


def fairness_group_mse_difference(
    y_true: Any,
    y_pred: Any,
    data: DataConfig | None = None,
    **kwargs: Any,
) -> float:
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


@dataclass(eq=False)
class DefaultFairlearnScoreConfig(_TaskAwareScorerMixin, FairlearnScoreDictConfig):
    """Default fairness scorer family with optional task inheritance."""

    classifier: Union[bool, str, None] = None
    scorers: dict[str, ScorerConfig] = field(default_factory=dict)

    def _build_default_scorers(
        self,
        classifier: bool,
    ) -> dict[str, ScorerConfig]:
        if classifier:
            return {
                "demographic_parity_difference": ScorerConfig(
                    score_name="demographic_parity_difference",
                    score_function="deckard.score.fairness.fairness_demographic_parity_difference",
                    greater_is_better=False,
                    needs_proba=False,
                ),
                "equalized_odds_difference": ScorerConfig(
                    score_name="equalized_odds_difference",
                    score_function="deckard.score.fairness.fairness_equalized_odds_difference",
                    greater_is_better=False,
                    needs_proba=False,
                ),
                "accuracy" : ScorerConfig(
                    score_name="accuracy",
                    score_function="sklearn.metrics.accuracy_score",
                    greater_is_better=True,
                    needs_proba=False,
                ),
                "precision" : ScorerConfig(
                    score_name="precision",
                    score_function="sklearn.metrics.precision_score",
                    greater_is_better=True,
                    needs_proba=False,
                    score_params = {"average": "weighted", "zero_division" : 0}
                ),
                "recall" : ScorerConfig(
                    score_name="recall",
                    score_function="sklearn.metrics.recall_score",
                    greater_is_better=True,
                    needs_proba=False,
                    score_params = {"average": "weighted", "zero_division" : 0}
                ),
                "f1" : ScorerConfig(
                    score_name="f1",
                    score_function="sklearn.metrics.f1_score",
                    greater_is_better=True,
                    needs_proba=False,
                    score_params = {"average": "weighted", "zero_division" : 0}
                ),
                
            }
        else:
            return {
            "mse": ScorerConfig(
                score_name="mse",
                score_function="sklearn.metrics.mean_squared_error",
                greater_is_better=False,
                needs_proba=False,
            ),
            "rmse": ScorerConfig(
                score_name="rmse",
                score_function="sklearn.metrics.mean_squared_error",
                greater_is_better=False,
                needs_proba=False,
                score_params={
                    "squared": False,
                },
            ),
            "mae": ScorerConfig(
                score_name="mae",
                score_function="sklearn.metrics.mean_absolute_error",
                greater_is_better=False,
                needs_proba=False,
            ),
            "r2": ScorerConfig(
                score_name="r2",
                score_function="sklearn.metrics.r2_score",
                greater_is_better=True,
                needs_proba=False,
            ),
            "group_mean_prediction_difference": ScorerConfig(
                score_name="group_mean_prediction_difference",
                score_function=(
                    "deckard.score.fairness_group_mean_prediction_difference"
                ),
                greater_is_better=False,
                needs_proba=False,
            ),
            "group_mae_difference": ScorerConfig(
                score_name="group_mae_difference",
                score_function=(
                    "deckard.score.fairness_group_mae_difference"
                ),
                greater_is_better=False,
                needs_proba=False,
            ),
            "group_mse_difference": ScorerConfig(
                score_name="group_mse_difference",
                score_function=(
                    "deckard.score.fairness_group_mse_difference"
                ),
                greater_is_better=False,
                needs_proba=False,
            ),
        }
            
            
        
            

    def _build_default_group_scorers(self, classifier: bool) -> dict[str, ScorerConfig]:
        if classifier:
            return {}
        # For regression, group metrics go here
        return {
            "group_mean_prediction_difference": ScorerConfig(
                score_name="group_mean_prediction_difference",
                score_function="deckard.score.fairness.fairness_group_mean_prediction_difference",
                greater_is_better=False,
                needs_proba=False,
            ),
            "group_mae_difference": ScorerConfig(
                score_name="group_mae_difference",
                score_function="deckard.score.fairness.fairness_group_mae_difference",
                greater_is_better=False,
                needs_proba=False,
            ),
            "group_mse_difference": ScorerConfig(
                score_name="group_mse_difference",
                score_function="deckard.score.fairness.fairness_group_mse_difference",
                greater_is_better=False,
                needs_proba=False,
            ),
        }

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        # Ensure scorers is always populated for group scoring
        if not getattr(self, 'scorers', None):
            self.scorers = self._build_default_scorers(bool(self.classifier))
        # Always enable group scoring by default if not explicitly set
        if not getattr(self, 'group_scorers', None):
            self.group_scorers = cast(Any, self._build_default_group_scorers(bool(self.classifier)))
        super().__post_init__()


@dataclass(eq=False)
class DefaultFairlearnClassificationConfig(DefaultFairlearnScoreConfig):
    """Default scorer set for classification fairness workflows."""

    classifier: Union[bool, str, None] = True


@dataclass(eq=False)
class DefaultFairlearnRegressionConfig(DefaultFairlearnScoreConfig):
    """Default scorer set for regression fairness workflows."""

    classifier: Union[bool, str, None] = False




safe_store(
    group="score",
    name="fairlearn-classification",
    node={"_target_": "deckard.score.fairness.DefaultFairlearnScoreConfig", "classifier": True},
)
safe_store(
    group="score",
    name="fairlearn-regression",
    node={"_target_": "deckard.score.fairness.DefaultFairlearnScoreConfig", "classifier": False},
)
