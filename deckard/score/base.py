from __future__ import annotations

"""Core scoring primitives and default scorer profiles."""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Union, cast

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..types import ArrayLike, AttackLike, EstimatorLike, MatrixLike
from ..utils import (
    BaseConfig,
    coerce_config,
    is_default_config_value,
    is_null_config_value,
    load_class,
    merge_list_of_dicts,
    resolve_class,
    safe_store,
)
from .canon import (
    DEFAULT_SCORING_MODE_BY_TYPE,
    DEFAULT_SCORING_STAGE_BY_TYPE,
    SCORING_STAGE_TOKEN_ALIASES,
    SUPPORTED_ATTACK_SCORE_MODES,
    SUPPORTED_DATA_SCORE_MODES,
    SUPPORTED_DETECTOR_SCORE_MODES,
    SUPPORTED_EXPERIMENT_DEFENSE_SCORING_STAGES,
    SUPPORTED_EXPERIMENT_SCORE_MODES,
    SUPPORTED_MODEL_SCORE_MODES,
    SUPPORTED_PIPELINE_SCORE_MODES,
    SUPPORTED_SCORING_STAGES,
    ScoringAttackStage,
    ScoringDataStage,
    ScoringDefenseStage,
    ScoringDetectorStage,
    ScoringModelStage,
    ScoringPipelineStage,
    normalize_scorer_mode,
)
from ._runtime import series_like_to_float_dict as _series_like_to_float_dict

logger = logging.getLogger(__name__)


def to_numpy_if_torch(value: Any) -> Any:
    """Recursively convert tensor-like payloads to numpy arrays when possible."""
    if isinstance(value, list):
        return [to_numpy_if_torch(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_numpy_if_torch(v) for v in value)
    if all(hasattr(value, attr) for attr in ("detach", "cpu", "numpy")):
        try:
            return value.detach().cpu().numpy()
        except Exception:
            return value
    return value


def normalize_scoring_mode(mode: str | None) -> str:
    """Normalize score mode tokens needed by core scorer feature resolution."""
    resolved = normalize_scorer_mode(mode)
    if resolved == "attack-val":
        return "val"
    return resolved


MetricScalar = Union[float, int, np.floating, np.integer]
MetricResult = Union[MetricScalar, np.ndarray]
ScoreFunction = Callable[..., MetricResult]
ScoreKwargValue = (
    str
    | int
    | float
    | bool
    | None
    | ArrayLike
    | MatrixLike
    | list[str]
    | tuple[str, ...]
    | dict[str, str | int | float | bool | None]
)


class _DataScorerMarker:
    """Mixin that marks a ScorerDictConfig as operating on data rather than model predictions.

    Inherit this class alongside ``ScorerDictConfig`` to signal that the scorer
    should be routed to ``data.scorer`` (rather than ``model.scorer``) when used
    in a score chain via :class:`~deckard.experiment.ExperimentConfig`.
    """


class _AttackProfileScorer:
    """Mixin that marks a ScorerDictConfig as an attack profile scorer.

    Subclasses must set ``_profile_attr`` to the :class:`AttackScorerConfig`
    attribute name (e.g. ``"evasion"``).  When used in a score chain, the scorer
    is applied to ``attack.scorer.<_profile_attr>`` rather than the model scorer.
    """

    _profile_attr: str = "evasion"


@dataclass(eq=False, kw_only=True)
class ScorerTypePlugin:
    """Generic scorer plugin that binds one mixin to one scoring family/subtype.

    Initialization fields
    ---------------------
    mixin_type : Any
        Mixin class (or import path) implementing runtime ``__call__``.
    scoring_type : str
        Scoring scope this plugin matches (e.g., "model", "data").
    scoring_subtype : str | None
        Optional subtype constraint (e.g., "classifier", "regressor", "fairness").
    excluded_subtypes : tuple[str, ...]
        Subtypes explicitly excluded from this plugin match.
    init_params : dict[str, Any]
        Metadata-only declaration payload for class/type/library docs.

    Plugin hooks
    ------------
    - ``resolve_scorer_mixins`` contributes mixins to runtime scorer context assembly.
    - ``resolve_scorer_handler`` returns callable handler for dispatch.
    - ``__call__`` forwards ``*args``/``**kwargs`` to the configured mixin instance
      bound to the runtime config.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    mixin_type: Any
    scoring_type: str
    scoring_subtype: Union[str, None] = None
    excluded_subtypes: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={
            "help": "Optional scoring subtypes that this plugin should not handle.",
        },
    )
    init_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Keyword arguments used when instantiating the scorer mixin or handler.",
        },
    )

    def _resolve_mixin_type(self) -> type:
        if isinstance(self.mixin_type, str):
            resolved = resolve_class(self.mixin_type)
            self.mixin_type = resolved
            return resolved
        return self.mixin_type

    def _matches(
        self,
        *,
        scoring_type: str,
        scoring_subtype: Union[str, None],
    ) -> bool:
        if (scoring_type or "").lower() != (self.scoring_type or "").lower():
            return False
        subtype = (scoring_subtype or "").lower()
        if (
            self.scoring_subtype is not None
            and subtype != self.scoring_subtype.lower()
        ):
            return False
        if subtype in {item.lower() for item in self.excluded_subtypes}:
            return False
        return True

    def resolve_scorer_mixins(
        self,
        runtime: "ScorerDictConfig",
        *,
        scoring_type: str,
        scoring_subtype: Union[str, None],
        default_mixins: tuple[type, ...],
    ) -> tuple[type, ...]:
        """Return mixin tuple for matching scoring family/subtype.

        Args:
            runtime: Active runtime scorer config.
            scoring_type: Requested scorer family.
            scoring_subtype: Requested scorer subtype.
            default_mixins: Default mixins for this scoring family.

        Returns:
            Mixin tuple to attach to runtime context.
        """
        _ = (runtime, default_mixins)
        if not self._matches(
            scoring_type=scoring_type,
            scoring_subtype=scoring_subtype,
        ):
            return ()
        mixin = self._resolve_mixin_type()
        return (mixin,)

    def resolve_scorer_handler(
        self,
        runtime: "ScorerDictConfig",
        *,
        scoring_type: str,
        scoring_subtype: Union[str, None],
        default_handler: Callable[..., ScoreDict] | None,
        default_mixins: tuple[type, ...],
    ) -> Callable[..., ScoreDict] | None:
        """Return callable runtime handler for matching scoring family/subtype.

        Args:
            runtime: Active runtime scorer config.
            scoring_type: Requested scorer family.
            scoring_subtype: Requested scorer subtype.
            default_handler: Existing resolved handler.
            default_mixins: Existing resolved mixins.

        Returns:
            Callable scorer handler when plugin matches; otherwise ``None``.
        """
        _ = (default_handler, default_mixins)
        if not self._matches(
            scoring_type=scoring_type,
            scoring_subtype=scoring_subtype,
        ):
            return None
        return lambda *args, **kwargs: self(runtime, *args, **kwargs)

    def __call__(self, runtime: "ScorerDictConfig", *args, **kwargs) -> ScoreDict:
        """Delegate runtime scorer execution to configured mixin handler.

        Args:
            runtime: Runtime config instance currently orchestrating scoring.
            *args: Positional runtime args forwarded to mixin ``__call__``.
            **kwargs: Keyword runtime args forwarded to mixin ``__call__``.

        Returns:
            Normalized score payload returned by the mixin handler.
        """
        mixin = self._resolve_mixin_type()
        handler = mixin(runtime)
        return ScoreDict.from_payload(handler(*args, **kwargs))


def _normalize_classifier_flag(
    classifier: Union[bool, str, None],
) -> Union[bool, None]:
    """Normalize classifier/regressor aliases to ``True`` / ``False`` / ``None``."""
    if classifier in ["classifier", True]:
        return True
    if classifier in ["regressor", False]:
        return False
    return None


@dataclass
class TaskAwareScorerMixin:
    """Mixin for scorer configs whose defaults depend on task type.

    API
    ---
    ``classifier``
        Optional explicit task selector. Accepted values are ``True``, ``False``,
        ``"classifier"``, ``"regressor"``, or ``None``.

    ``resolve_classifier(...)``
        Resolve the effective task from explicit config first, then runtime
        attack/model/data context, finally a caller-supplied default.

    ``_build_default_scorers(classifier)``
        Subclasses must return the default scorer mapping for the resolved task.

    ``_initialize_task_aware_scorers()``
        Populate ``self.scorers`` from ``_build_default_scorers`` when the user
        did not provide an explicit scorer mapping.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    classifier: Union[bool, str, None] = None

    def _normalize_classifier(self) -> None:
        self.classifier = _normalize_classifier_flag(getattr(self, "classifier", None))

    def resolve_classifier(
        self,
        *,
        data: "DataConfig | None" = None,
        model: EstimatorLike | None = None,
        attack: AttackLike | None = None,
        default: Union[bool, None] = None,
    ) -> bool:
        """Resolve the effective task type for this scorer config.

        Precedence is:
        1. explicit ``self.classifier``
        2. attack-derived runtime context
        3. ``model.classifier``
        4. ``data.classifier``
        5. explicit ``default``

        Args:
            data: Optional data runtime context.
            model: Optional model runtime context.
            attack: Optional attack runtime context.
            default: Fallback classifier flag when context is ambiguous.

        Returns:
            Resolved classifier flag where True means classification and False regression.

        Raises:
            ValueError: If classifier cannot be inferred from explicit/default/context values.
        """
        explicit = _normalize_classifier_flag(getattr(self, "classifier", None))
        if explicit is not None:
            return explicit

        if attack is not None:
            attack_classifier = _normalize_classifier_flag(
                getattr(attack, "classifier", None),
            )
            if attack_classifier is not None:
                return attack_classifier
            if hasattr(attack, "_is_continuous"):
                return not bool(getattr(attack, "_is_continuous"))

        model_classifier = _normalize_classifier_flag(
            getattr(model, "classifier", None),
        )
        if model_classifier is not None:
            return model_classifier

        data_classifier = _normalize_classifier_flag(
            getattr(data, "classifier", None),
        )
        if data_classifier is not None:
            return data_classifier

        if default is not None:
            return default
        raise ValueError(
            "Unable to resolve classifier/regression task for scorer config; "
            "set classifier explicitly or provide model/data/attack context.",
        )

    def _build_default_scorers(self, classifier: bool) -> dict[str, "ScorerConfig"]:
        raise NotImplementedError()

    def _initialize_task_aware_scorers(
        self,
        *,
        default: Union[bool, None] = None,
    ) -> None:
        self._normalize_classifier()
        if getattr(self, "scorers", None):
            return
        classifier = self.resolve_classifier(default=default)
        self.scorers = self._build_default_scorers(classifier=classifier)


@dataclass
class ScorerConfig:
    """Atomic scorer configuration.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    score_name: str
    score_function: Any
    score_params: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Keyword arguments forwarded to the scorer function during evaluation.",
        },
    )
    metric_scope: str = "auto"
    stage: List[str] = field(
        default_factory=list,
        metadata={"help": "Configuration field: stage."},
    )
    greater_is_better: bool = True
    needs_labels: Union[bool, None] = True
    needs_proba: Union[bool, None] = None
    needs_logits: Union[bool, None] = None
    binary_expand_to_multiclass: Union[bool, None] = None
    binary_positive_class_index: int = 1
    row_sum_atol: float = 1e-2
    probability_clip_eps: float = 1e-12

    def __post_init__(self):
        if OmegaConf.is_config(self.score_function):
            self.score_function = OmegaConf.to_container(
                self.score_function,
                resolve=True,
            )
        if isinstance(self.score_function, dict):
            score_fn_spec = {str(k): v for k, v in dict(self.score_function).items()}
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
        scope = str(self.metric_scope).strip().lower()
        if scope not in {"auto", "standard", "group", "reduced"}:
            raise ValueError(
                f"Scorer '{self.score_name}' metric_scope must be one of "
                "{'auto', 'standard', 'group', 'reduced'}",
            )
        self.metric_scope = scope
        if self.needs_labels is True and self.needs_proba is True:
            raise ValueError(
                f"Scorer '{self.score_name}' cannot set both needs_labels=True and needs_proba=True",
            )
        if self.binary_positive_class_index < 0:
            raise ValueError("binary_positive_class_index must be >= 0")
        if self.row_sum_atol <= 0:
            raise ValueError("row_sum_atol must be > 0")
        if self.probability_clip_eps <= 0:
            raise ValueError("probability_clip_eps must be > 0")
        if self.needs_labels is not None:
            self.needs_labels = bool(self.needs_labels)
        if self.needs_proba is not None:
            self.needs_proba = bool(self.needs_proba)
        if self.needs_logits is not None:
            self.needs_logits = bool(self.needs_logits)
        if self.binary_expand_to_multiclass is not None:
            self.binary_expand_to_multiclass = bool(self.binary_expand_to_multiclass)
        self.stage = ScorerDictConfig._normalize_stage_field(self.stage)

    def _validate_raw_output_input(self, dep, ind):
        dep_arr = np.asarray(to_numpy_if_torch(dep))
        ind_arr = np.asarray(to_numpy_if_torch(ind))
        if ind_arr.ndim not in (1, 2):
            raise ValueError(
                f"Raw-output scorer '{self.score_name}' requires 1D/2D input; got shape {ind_arr.shape}",
            )
        if ind_arr.shape[0] != dep_arr.shape[0]:
            raise ValueError(
                f"Raw-output scorer '{self.score_name}' requires matching sample counts; got {ind_arr.shape[0]} predictions for {dep_arr.shape[0]} labels",
            )
        if not np.issubdtype(ind_arr.dtype, np.number):
            raise ValueError(
                f"Raw-output scorer '{self.score_name}' requires numeric outputs; got dtype {ind_arr.dtype}",
            )

    def _looks_like_probabilities(self, ind_arr: np.ndarray) -> bool:
        if ind_arr.ndim == 1:
            return bool(np.nanmin(ind_arr) >= 0.0 and np.nanmax(ind_arr) <= 1.0)
        if ind_arr.ndim == 2:
            row_sums = np.sum(ind_arr, axis=1)
            return bool(
                np.nanmin(ind_arr) >= 0.0
                and np.nanmax(ind_arr) <= 1.0
                and np.allclose(row_sums, 1.0, atol=self.row_sum_atol),
            )
        return False

    def _convert_logits_if_needed(self, ind_arr: np.ndarray) -> np.ndarray:
        if self.needs_logits is not True:
            return ind_arr
        if self._looks_like_probabilities(ind_arr):
            return ind_arr
        if ind_arr.ndim == 1:
            return 1.0 / (1.0 + np.exp(-ind_arr))
        if ind_arr.ndim == 2:
            shifted = ind_arr - np.max(ind_arr, axis=1, keepdims=True)
            exp_logits = np.exp(shifted)
            denom = np.sum(exp_logits, axis=1, keepdims=True)
            return exp_logits / np.clip(denom, self.probability_clip_eps, None)
        return ind_arr

    def _binary_task_detected(self, dep_arr: np.ndarray) -> bool:
        if dep_arr.ndim != 1:
            return False
        return np.unique(dep_arr).size <= 2

    def _expand_binary_to_multiclass(self, ind_arr: np.ndarray) -> np.ndarray:
        if ind_arr.ndim == 1:
            p = ind_arr
            return np.column_stack([1.0 - p, p])
        if ind_arr.ndim == 2 and ind_arr.shape[1] == 1:
            p = ind_arr.reshape(-1)
            return np.column_stack([1.0 - p, p])
        return ind_arr

    def _normalize_predictions_for_metric(self, dep, ind):
        """Normalize probabilities/logits to metric-compatible labels when needed."""
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
            "demographic_parity_difference",
            "equalized_odds_difference",
        }
        is_label_metric = (
            metric_name in label_metrics or self.score_name in label_metrics
        )

        if self.needs_proba is True:
            # Expects raw model outputs (logits/probabilities)
            ind_arr = np.asarray(to_numpy_if_torch(ind))
            self._validate_raw_output_input(dep=dep, ind=ind_arr)
            dep_arr = np.asarray(to_numpy_if_torch(dep))
            ind_arr = self._convert_logits_if_needed(ind_arr)

            should_expand_binary = (
                self.binary_expand_to_multiclass
                if self.binary_expand_to_multiclass is not None
                else True
            )
            if should_expand_binary and self._binary_task_detected(dep_arr):
                ind_arr = self._expand_binary_to_multiclass(ind_arr)

            score_name = str(self.score_name).lower()
            metric_name_l = str(metric_name).lower()
            is_roc_auc_metric = metric_name_l == "roc_auc_score" or score_name in {
                "roc_auc",
                "roc_auc_score",
            }
            is_log_loss_metric = metric_name_l == "log_loss" or score_name in {
                "log_loss",
                "logloss",
            }
            if is_roc_auc_metric and ind_arr.ndim == 2:
                if ind_arr.shape[1] == 1:
                    return ind_arr.reshape(-1)
                # sklearn binary roc_auc_score expects 1D positive-class scores.
                if dep_arr.ndim == 1 and ind_arr.shape[1] == 2:
                    unique_labels = np.unique(dep_arr)
                    if unique_labels.size <= 2:
                        index = min(
                            self.binary_positive_class_index,
                            ind_arr.shape[1] - 1,
                        )
                        return ind_arr[:, index]
            if is_log_loss_metric:
                if ind_arr.ndim == 2 and ind_arr.shape[1] == 1:
                    return ind_arr.reshape(-1)
            return ind_arr

        if self.needs_labels is not True:
            return ind

        if not is_label_metric:
            return ind

        dep_arr = np.asarray(to_numpy_if_torch(dep))
        ind_arr = np.asarray(to_numpy_if_torch(ind))
        if dep_arr.ndim != 1 or ind_arr.ndim != 2:
            return ind
        if not np.issubdtype(ind_arr.dtype, np.number):
            return ind
        if ind_arr.shape[1] == 1:
            binary_scores = ind_arr.reshape(-1)
            threshold = 0.5
            if np.nanmin(binary_scores) < 0.0 or np.nanmax(binary_scores) > 1.0:
                threshold = 0.0
            return (binary_scores >= threshold).astype(int)
        return np.argmax(ind_arr, axis=1)

    def __call__(
        self,
        dep: ArrayLike | MatrixLike | None = None,
        ind: ArrayLike | MatrixLike | None = None,
        swap: bool = False,
        **kwargs: ScoreKwargValue,
    ) -> MetricResult:
        """Execute one scorer using generic dependent/independent payload names.

        Args:
            dep: Dependent payload (typically labels or targets).
            ind: Independent payload (typically predictions or features).
            swap: Swap ``dep``/``ind`` before scoring.
            **kwargs: Additional metric kwargs.

        Returns:
            Metric output returned by the configured score function.

        Raises:
            ValueError: If required dep/ind inputs are missing.
            TypeError: If score_function is not callable.
        """
        if dep is None and "y" in kwargs:
            dep = kwargs.pop("y")
        if dep is None and "y_true" in kwargs:
            dep = kwargs.pop("y_true")
        if ind is None and "X" in kwargs:
            ind = kwargs.pop("X")
        if ind is None and "y_pred" in kwargs:
            ind = kwargs.pop("y_pred")
        if dep is None or ind is None:
            raise ValueError("ScorerConfig requires both dep and ind inputs")

        if swap:
            dep, ind = ind, dep

        dep = to_numpy_if_torch(dep)
        ind = to_numpy_if_torch(ind)
        ind = self._normalize_predictions_for_metric(dep=dep, ind=ind)

        params = {**self.score_params, **kwargs}
        for reserved in ("y_true", "y_pred", "dep", "ind"):
            params.pop(reserved, None)
        score_function = self.score_function
        if not callable(score_function):
            raise TypeError(
                "score_function must be callable after ScorerConfig initialization",
            )

        signature = inspect.signature(score_function)
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
            accepted.discard("dep")
            accepted.discard("ind")
            params = {k: v for k, v in params.items() if k in accepted}
        return cast(MetricResult, score_function(dep, ind, **params))


@dataclass(eq=False, kw_only=True)
class ScorerDictConfig(BaseConfig):
    """Container of named ScorerConfig instances.

    Attributes
    ----------
    scorers : dict[str, ScorerConfig]
        Mapping of scorer name to ScorerConfig instances.
    stage : list[str], optional
        Generic stage selector for this scorer profile. Supports a single stage
        token (e.g., ``"test"``) or multiple stage tokens
        (e.g., ``["post-defense", "post-pipeline"]``).

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    scorers: dict[str, ScorerConfig] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: scorers."},
    )
    stage: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Optional runtime stages where this scorer is allowed to execute.",
        },
    )

    def _is_data_profile_scorer(self) -> bool:
        if isinstance(self, _DataScorerMarker):
            return True
        return str(getattr(self, "scoring_type", "")).strip().lower() == "data"

    def __post_init__(self):
        self.stage = self._normalize_stage_field(self.stage)

        normalized = {}
        for key, value in self.scorers.items():
            if isinstance(value, ScorerConfig):
                scorer = value
            elif isinstance(value, dict):
                scorer_data = dict(value)
                raw_score_name = scorer_data.pop("score_name", key)
                raw_score_params = scorer_data.pop("score_params", {})
                raw_metric_scope = scorer_data.pop("metric_scope", "auto")
                raw_stage = scorer_data.pop("stage", "")
                raw_needs_labels = scorer_data.pop("needs_labels", None)
                raw_needs_proba = scorer_data.pop("needs_proba", None)
                raw_needs_logits = scorer_data.pop("needs_logits", None)
                raw_binary_expand = scorer_data.pop(
                    "binary_expand_to_multiclass",
                    None,
                )
                raw_positive_idx = scorer_data.pop("binary_positive_class_index", 1)
                raw_row_sum_atol = scorer_data.pop("row_sum_atol", 1e-2)
                raw_prob_clip_eps = scorer_data.pop("probability_clip_eps", 1e-12)
                resolved_needs_labels = (
                    True
                    if raw_needs_labels is None and raw_needs_proba is not True
                    else (
                        False if raw_needs_labels is None else bool(raw_needs_labels)
                    )
                )
                resolved_needs_proba = (
                    bool(raw_needs_proba) if raw_needs_proba is not None else None
                )
                if not isinstance(raw_score_params, dict):
                    raise TypeError(
                        f"score_params for '{key}' must be a dictionary, got {type(raw_score_params)}",
                    )
                scorer = ScorerConfig(
                    score_name=str(raw_score_name),
                    score_function=scorer_data.pop("score_function"),
                    score_params=dict(raw_score_params),
                    metric_scope=str(raw_metric_scope),
                    stage=self._normalize_stage_field(raw_stage),
                    greater_is_better=bool(
                        scorer_data.pop(
                            "greater_is_better",
                            True,
                        ),
                    ),
                    needs_labels=resolved_needs_labels,
                    needs_proba=resolved_needs_proba,
                    needs_logits=(
                        bool(raw_needs_logits)
                        if raw_needs_logits is not None
                        else None
                    ),
                    binary_expand_to_multiclass=(
                        bool(raw_binary_expand)
                        if raw_binary_expand is not None
                        else None
                    ),
                    binary_positive_class_index=int(raw_positive_idx),
                    row_sum_atol=float(raw_row_sum_atol),
                    probability_clip_eps=float(raw_prob_clip_eps),
                )
            elif isinstance(value, DictConfig):
                raw_value = OmegaConf.to_container(value, resolve=True)
                if not isinstance(raw_value, dict):
                    raise TypeError(
                        f"DictConfig scorer entry '{key}' must resolve to a dictionary, got {type(raw_value)}",
                    )
                scorer_data = dict(raw_value)
                raw_score_name = scorer_data.pop("score_name", key)
                raw_score_params = scorer_data.pop("score_params", {})
                raw_metric_scope = scorer_data.pop("metric_scope", "auto")
                raw_stage = scorer_data.pop("stage", "")
                raw_needs_labels = scorer_data.pop("needs_labels", None)
                raw_needs_proba = scorer_data.pop("needs_proba", None)
                raw_needs_logits = scorer_data.pop("needs_logits", None)
                raw_binary_expand = scorer_data.pop(
                    "binary_expand_to_multiclass",
                    None,
                )
                raw_positive_idx = scorer_data.pop("binary_positive_class_index", 1)
                raw_row_sum_atol = scorer_data.pop("row_sum_atol", 1e-2)
                raw_prob_clip_eps = scorer_data.pop("probability_clip_eps", 1e-12)
                resolved_needs_labels = (
                    True
                    if raw_needs_labels is None and raw_needs_proba is not True
                    else (
                        False if raw_needs_labels is None else bool(raw_needs_labels)
                    )
                )
                resolved_needs_proba = (
                    bool(raw_needs_proba) if raw_needs_proba is not None else None
                )
                if not isinstance(raw_score_params, dict):
                    raise TypeError(
                        f"score_params for '{key}' must be a dictionary, got {type(raw_score_params)}",
                    )
                scorer = ScorerConfig(
                    score_name=str(raw_score_name),
                    score_function=scorer_data.pop("score_function"),
                    score_params=dict(raw_score_params),
                    metric_scope=str(raw_metric_scope),
                    stage=self._normalize_stage_field(raw_stage),
                    greater_is_better=bool(
                        scorer_data.pop(
                            "greater_is_better",
                            True,
                        ),
                    ),
                    needs_labels=resolved_needs_labels,
                    needs_proba=resolved_needs_proba,
                    needs_logits=(
                        bool(raw_needs_logits)
                        if raw_needs_logits is not None
                        else None
                    ),
                    binary_expand_to_multiclass=(
                        bool(raw_binary_expand)
                        if raw_binary_expand is not None
                        else None
                    ),
                    binary_positive_class_index=int(raw_positive_idx),
                    row_sum_atol=float(raw_row_sum_atol),
                    probability_clip_eps=float(raw_prob_clip_eps),
                )
            else:
                raise TypeError(
                    f"Value for key '{key}' must be ScorerConfig or dict, got {type(value)}",
                )
            scorer.stage = self._normalize_stage_field(getattr(scorer, "stage", ""))
            normalized[key] = scorer
        self.scorers = normalized
        self._validate_scope_mode_compatibility()

    def _validate_scope_mode_compatibility(self) -> None:
        """Fail fast on scorer-scope/mode combinations that are semantically invalid."""
        scorer_is_data_profile = self._is_data_profile_scorer()

        scoring_type = str(getattr(self, "scoring_type", "")).strip().lower()
        if scoring_type not in {
            "",
            "data",
            "model",
            "attack",
            "detector",
            "experiment",
        }:
            raise ValueError(
                f"Unsupported scoring_type '{scoring_type}'.",
            )

        container_tokens = self._stage_tokens(self.stage)
        if (
            not scorer_is_data_profile
        ) and ScoringDataStage.PRE_SAMPLE.value in container_tokens:
            raise ValueError(
                "pre-sample stage is reserved for data-profile scorers.",
            )

        for key, scorer in self.scorers.items():
            scorer_tokens = self._stage_tokens(getattr(scorer, "stage", ""))

            if scorer_is_data_profile and scorer.needs_proba is True:
                raise ValueError(
                    f"Data scorer '{key}' cannot set needs_proba=True; "
                    "data-profile scorers operate on X/y data splits, not model probability outputs.",
                )

            if (
                not scorer_is_data_profile
            ) and ScoringDataStage.PRE_SAMPLE.value in scorer_tokens:
                raise ValueError(
                    f"Scorer '{key}' declares pre-sample stage but is not a data-profile scorer.",
                )

    @staticmethod
    def _normalize_stage_token(token: str) -> str:
        normalized = str(token).strip().lower()
        return SCORING_STAGE_TOKEN_ALIASES.get(normalized, normalized)

    @staticmethod
    def _normalize_stage_field(
        stage_value: Union[str, list[str], tuple[str, ...], None],
    ) -> List[str]:
        if stage_value is None:
            return []
        if isinstance(stage_value, str):
            normalized_text = stage_value.strip().lower()
            ScorerDictConfig._validate_stage_tokens(
                ScorerDictConfig._stage_tokens(normalized_text),
                field_name="stage",
            )
            return [normalized_text] if normalized_text != "" else []
        if isinstance(stage_value, (tuple, ListConfig)):
            stage_value = list(stage_value)
        if isinstance(stage_value, list):
            normalized: list[str] = []
            for item in stage_value:
                text = str(item).strip().lower()
                if text != "":
                    normalized.append(text)
            ScorerDictConfig._validate_stage_tokens(
                ScorerDictConfig._stage_tokens(normalized),
                field_name="stage",
            )
            return normalized
        normalized_text = str(stage_value).strip().lower()
        ScorerDictConfig._validate_stage_tokens(
            ScorerDictConfig._stage_tokens(normalized_text),
            field_name="stage",
        )
        return [normalized_text] if normalized_text != "" else []

    @staticmethod
    def _validate_stage_tokens(stage_tokens: set[str], field_name: str) -> None:
        if len(stage_tokens) == 0:
            return
        unsupported = sorted(stage_tokens - SUPPORTED_SCORING_STAGES)
        if unsupported:
            raise ValueError(
                f"Unsupported {field_name} token(s): {unsupported}. "
                f"Supported stages: {sorted(SUPPORTED_SCORING_STAGES)}",
            )

    @staticmethod
    def _stage_tokens(stage_value: Any) -> set[str]:
        if stage_value is None:
            return set()
        if isinstance(stage_value, Enum):
            return {
                ScorerDictConfig._normalize_stage_token(
                    str(stage_value.value).strip().lower(),
                ),
            }
        if isinstance(stage_value, str):
            tokens = [
                ScorerDictConfig._normalize_stage_token(token.strip().lower())
                for token in stage_value.split(",")
            ]
            return {token for token in tokens if token != ""}
        if isinstance(stage_value, (list, tuple, set, ListConfig)):
            merged: set[str] = set()
            for item in stage_value:
                merged.update(ScorerDictConfig._stage_tokens(item))
            return merged
        return {
            ScorerDictConfig._normalize_stage_token(
                str(stage_value).strip().lower(),
            ),
        }

    @classmethod
    def _runtime_stage_tokens(
        cls,
        *,
        mode: str | None,
        stage: Union[str, list[str], None] = None,
    ) -> set[str]:
        mode_token = "" if mode is None else str(mode).strip().lower()
        tokens: set[str] = set()
        valid_modes = set(SUPPORTED_SCORING_STAGES) | {
            "all",
            "attack",
            "attack-val",
            "",
        }
        if mode_token not in valid_modes:
            raise KeyError(
                f"Unsupported scoring mode '{mode}'. Expected one of: {sorted(valid_modes - {''})}",
            )
        if mode_token:
            tokens.add(mode_token)

        stage_tokens = cls._stage_tokens(stage)
        cls._validate_stage_tokens(stage_tokens, field_name="runtime stage")

        mode_aliases: dict[str, set[str]] = {
            "train": {
                ScoringModelStage.MODEL_TRAIN.value,
            },
            "test": {
                ScoringModelStage.MODEL_TEST.value,
                ScoringAttackStage.PRE_ATTACK.value,
            },
            "val": {
                ScoringModelStage.MODEL_VAL.value,
                ScoringDefenseStage.VAL_DEFENSE.value,
                ScoringPipelineStage.VAL_PIPELINE.value,
                ScoringAttackStage.VAL_ATTACK.value,
                ScoringDataStage.VAL_ATTACK.value,
                ScoringDetectorStage.VAL_FILTER.value,
            },
            "all": {
                ScoringModelStage.MODEL_TRAIN.value,
                ScoringModelStage.MODEL_TEST.value,
                ScoringModelStage.MODEL_VAL.value,
                ScoringAttackStage.PRE_ATTACK.value,
                ScoringDefenseStage.VAL_DEFENSE.value,
                ScoringPipelineStage.VAL_PIPELINE.value,
                ScoringAttackStage.VAL_ATTACK.value,
                ScoringDataStage.VAL_ATTACK.value,
                ScoringDetectorStage.VAL_FILTER.value,
            },
            "attack": {
                ScoringAttackStage.POST_ATTACK.value,
            },
            "attack-val": {
                ScoringAttackStage.POST_ATTACK.value,
                ScoringAttackStage.VAL_ATTACK.value,
                ScoringModelStage.MODEL_VAL.value,
                ScoringDefenseStage.VAL_DEFENSE.value,
                ScoringPipelineStage.VAL_PIPELINE.value,
                ScoringDataStage.VAL_ATTACK.value,
                ScoringDetectorStage.VAL_FILTER.value,
            },
            "pre-sample": {
                ScoringDataStage.PRE_SAMPLE.value,
            },
        }

        tokens.update(mode_aliases.get(mode_token, set()))
        tokens.update(stage_tokens)
        return tokens

    @classmethod
    def _stage_matches(
        cls,
        configured_stage: Any,
        runtime_stage_tokens: set[str],
    ) -> bool:
        configured = cls._stage_tokens(configured_stage)
        if len(configured) == 0:
            return True
        if len(runtime_stage_tokens) == 0:
            return False
        return not configured.isdisjoint(runtime_stage_tokens)

    def _default_runtime_mode(self) -> str:
        scoring_type = str(getattr(self, "scoring_type", "")).strip().lower()
        return DEFAULT_SCORING_MODE_BY_TYPE.get(scoring_type, "test")

    def _default_runtime_stage(self) -> str:
        scoring_type = str(getattr(self, "scoring_type", "")).strip().lower()
        return DEFAULT_SCORING_STAGE_BY_TYPE.get(scoring_type, "test")

    @staticmethod
    def _first_stage_value(requested_stage: Union[str, list[str], None]) -> str | None:
        if requested_stage is None:
            return None
        if isinstance(requested_stage, str):
            token = requested_stage.split(",", 1)[0].strip().lower()
            return token if token != "" else None
        if isinstance(requested_stage, (list, tuple, ListConfig)):
            for item in requested_stage:
                token = str(item).strip().lower()
                if token != "":
                    return token
            return None
        token = str(requested_stage).strip().lower()
        return token if token != "" else None

    def _resolve_stage_key(
        self,
        mode: str | None,
        requested_stage: Union[str, list[str], None] = None,
    ) -> str:
        if mode is not None:
            return self._resolve_runtime_mode(
                mode,
                requested_stage=requested_stage,
            )
        stage_value = self._first_stage_value(requested_stage)
        if stage_value is not None:
            return stage_value
        return self._default_runtime_stage()

    def _resolve_runtime_mode(
        self,
        mode: str | None,
        requested_stage: Union[str, list[str], None] = None,
    ) -> str:
        _ = requested_stage
        if mode is None:
            return normalize_scorer_mode(self._default_runtime_mode())
        return normalize_scorer_mode(mode)

    def __iter__(self):
        return iter(self.scorers.items())

    def __hash__(self):
        return super().__hash__()

    def __getitem__(self, key):
        return self.scorers[key]

    @property
    def configured_scorers(self) -> dict[str, ScorerConfig]:
        """Public accessor for configured scorer definitions.

        Returns:
            Mapping from scorer name to configured scorer object.
        """
        return self.scorers

    @configured_scorers.setter
    def configured_scorers(self, value: dict[str, ScorerConfig] | None) -> None:
        """Set configured scorer definitions.

        Args:
            value: Replacement scorer mapping.
        """
        self.scorers = value or {}

    def get_callables(self) -> dict[str, ScorerConfig]:
        """Return configured scorer callables keyed by scorer name.

        Returns:
            Mapping of scorer names to scorer configs.
        """
        return {key: scorer for key, scorer in self.scorers.items()}

    def score(
        self,
        ind: MatrixLike,
        dep: ArrayLike,
        *args: Any,
        data: "DataConfig | None" = None,
        model: EstimatorLike | None = None,
        attack: AttackLike | None = None,
        **kwargs: Any,
    ) -> ScoreDict:
        """Compute metrics from matrix-like independent and array-like dependent data.

        Args:
            ind: Matrix-like independent/reference payload.
            dep: Array-like dependent/target payload.
            *args: Additional positional runtime scoring payloads.
            data: Optional data context.
            model: Optional model context.
            attack: Optional attack context.
            **kwargs: Additional keyword scoring payloads. Supports `mode`.

        Returns:
            Metric outputs keyed by score name.
        """
        _ = args
        mode = kwargs.pop("mode", "test")
        return self.__call__(
            mode=mode,
            data=data,
            model=model,
            attack=attack,
            ind=ind,
            dep=dep,
            **kwargs,
        )

    @classmethod
    def merge(
        cls,
        items: list["ScorerDictConfig | dict[str, ScoreKwargValue]"],
    ) -> "ScorerDictConfig":
        """Merge a list of scorer specs into a single ScorerDictConfig.

        Each element of *items* may be a :class:`ScorerDictConfig`, a dict
        with a ``scorers`` key, or a bare scorers dict (name -> scorer spec).
        Later entries win on duplicate scorer names.

        Args:
            items: Scorer containers to merge.

        Returns:
            Consolidated scorer dictionary config.
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
    def resolve_mode_features(
        mode: str | None,
        data: "DataConfig | None",
    ) -> MatrixLike | None:
        """Resolve split-specific feature payload for the requested scoring mode.

        Args:
            mode: Requested scoring mode token.
            data: Optional data runtime context.

        Returns:
            Resolved feature payload for the mode, if available.
        """
        if data is None:
            return None
        try:
            resolved_mode = normalize_scoring_mode(mode)
        except ValueError:
            return None
        if resolved_mode == "train":
            return getattr(data, "X_train", None)
        if resolved_mode == "test":
            return getattr(data, "X_test", None)
        if resolved_mode == "val":
            return getattr(data, "X_val", None)
        if resolved_mode in {"pre-sample", "all"}:
            return getattr(data, "X", getattr(data, "_X", None))
        return None

    @staticmethod
    def is_classification_labels(y: ArrayLike | MatrixLike) -> bool:
        """Return ``True`` when label payload appears categorical/integer-coded.

        Args:
            y: Label payload to inspect.

        Returns:
            ``True`` when labels appear categorical/integer-coded.
        """
        # Returns True if y is integer/binary labels, False if continuous
        y_arr = np.asarray(to_numpy_if_torch(y))
        if y_arr.dtype.kind in {"i", "u", "b"}:
            return True
        # Heuristic: if all values are 0/1 or small integer classes
        if np.issubdtype(y_arr.dtype, np.number):
            unique = np.unique(y_arr)
            if len(unique) <= 20 and np.all(np.equal(np.mod(unique, 1), 0)):
                return True
        return False

    @staticmethod
    def predict_proba_from_model(
        model: EstimatorLike | None,
        X: MatrixLike | None,
        y_true: ArrayLike | None = None,
        y_pred: ArrayLike | MatrixLike | None = None,
    ) -> MatrixLike | ArrayLike:
        """Resolve probability-like prediction outputs from model runtime.

        Args:
            model: Model runtime/config context.
            X: Input features for inference.
            y_true: Optional labels used for fallback one-hot conversion.
            y_pred: Optional precomputed predictions/probabilities.

        Returns:
            Probability-like prediction payload.

        Raises:
            ValueError: If probability outputs cannot be derived from model or inputs.
        """
        if model is None or X is None:
            raise ValueError("Cannot compute probabilities: model or input X is None.")

        estimator = None
        if hasattr(model, "get_model") and callable(model.get_model):
            try:
                estimator = model.get_model()
            except Exception:
                estimator = None
        if estimator is None:
            estimator = getattr(model, "model", getattr(model, "_model", None))

        # Try predict_proba or _predict_proba on the model
        for proba_method in ("predict_proba", "_predict_proba"):
            predict_proba = getattr(model, proba_method, None)
            if callable(predict_proba):
                return predict_proba(X)
        # Try estimator if available
        estimator = getattr(model, "model", getattr(model, "_model", None))
        if estimator is not None:
            for proba_method in ("predict_proba", "_predict_proba"):
                predict_proba = getattr(estimator, proba_method, None)
                if callable(predict_proba):
                    return predict_proba(X)
        # Fallback: try predict or _predict
        predict_fn = getattr(model, "predict", getattr(model, "_predict", None))
        if not callable(predict_fn):
            raise ValueError(
                "Model must have a predict or predict_proba function for probability metrics.",
            )
        # If y_pred is provided and looks like probabilities, use it
        if y_pred is not None:
            arr = np.asarray(y_pred)
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                # Heuristic: if all values in [0,1] or row sums ~1, treat as proba
                if np.all((arr >= 0) & (arr <= 1)) and np.allclose(
                    arr.sum(axis=1),
                    1,
                    atol=1e-2,
                ):
                    return arr
            # Fallback: if arr is 1D class labels and y_true is available, convert to one-hot
            if arr.ndim == 1 and y_true is not None:
                import warnings

                warnings.warn(
                    "Probability scorer received class labels instead of probabilities; converting to one-hot encoding as fallback.",
                )
                y_true_arr = np.asarray(y_true)
                classes = np.unique(y_true_arr)
                n_classes = len(classes)
                # Map labels to indices in classes
                class_to_index = {c: i for i, c in enumerate(classes)}
                one_hot = np.zeros((arr.shape[0], n_classes), dtype=float)
                for i, label in enumerate(arr):
                    idx = class_to_index.get(label, None)
                    if idx is not None:
                        one_hot[i, idx] = 1.0
                return one_hot
            # Otherwise, raise error
            raise ValueError(
                "Probability scorer requires probability outputs (1D/2D array of probabilities), but got class labels or invalid shape.",
            )
        raise ValueError(
            "Probability scorer requires probability outputs, but model does not support predict_proba and y_pred is not a valid probability array.",
        )

    @staticmethod
    def _resolve_ind_dep_from_kwargs(
        ind: MatrixLike | ArrayLike | None,
        dep: MatrixLike | ArrayLike | None,
        kwargs: dict[str, Any],
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        if ind is None and "X" in kwargs:
            ind = kwargs.pop("X")
        if ind is None and "y_pred" in kwargs:
            ind = kwargs.pop("y_pred")
        if dep is None and "y" in kwargs:
            dep = kwargs.pop("y")
        if dep is None and "y_true" in kwargs:
            dep = kwargs.pop("y_true")
        return ind, dep

    def _resolve_mode_payload(
        self,
        effective_mode: str,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        attack: AttackLike | None,
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        resolver_map = {
            "test": self._mode_payload_test,
            "train": self._mode_payload_train,
            "attack": self._mode_payload_attack,
            "val": self._mode_payload_val,
            "all": self._mode_payload_all,
            "attack-val": self._mode_payload_attack_val,
            "pre-sample": self._mode_payload_pre_sample,
        }
        resolver = resolver_map.get(effective_mode)
        if resolver is None:
            return None, None
        return resolver(data=data, model=model, attack=attack)

    @staticmethod
    def _mode_payload_test(
        *,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        attack: AttackLike | None,
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        del attack
        assert data is not None
        dep = data.y_test
        ind = getattr(model, "test_predictions", None)
        if ind is None:
            ind = getattr(model, "predictions", None)
        return ind, dep

    @staticmethod
    def _mode_payload_train(
        *,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        attack: AttackLike | None,
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        del attack
        assert data is not None and model is not None
        return model.training_predictions, data.y_train

    @staticmethod
    def _mode_payload_attack(
        *,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        attack: AttackLike | None,
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        del model
        assert data is not None and attack is not None
        dep = getattr(attack, "attacked_labels", None)
        if dep is None:
            y_test = getattr(data, "y_test", None)
            if y_test is None:
                raise ValueError(
                    "attack mode requires attack.attacked_labels or data.y_test",
                )
            dep = y_test[: attack.attack_size]
        return attack.attack_predictions, dep

    @staticmethod
    def _mode_payload_val(
        *,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        attack: AttackLike | None,
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        del attack
        assert data is not None and model is not None
        return model.val_predictions, data.y_val

    @staticmethod
    def _mode_payload_all(
        *,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        attack: AttackLike | None,
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        del attack
        assert data is not None and model is not None
        dep = getattr(data, "_y", None)
        if dep is None:
            y_parts = [
                getattr(data, "y_train", None),
                getattr(data, "y_test", None),
                getattr(data, "y_val", None),
            ]
            y_parts = [part for part in y_parts if part is not None]
            if len(y_parts) > 0:
                dep = np.concatenate([np.asarray(part) for part in y_parts])

        ind = getattr(model, "predictions", None)
        if ind is None:
            pred_parts = [
                getattr(model, "training_predictions", None),
                getattr(model, "test_predictions", None),
                getattr(model, "val_predictions", None),
            ]
            pred_parts = [part for part in pred_parts if part is not None]
            if len(pred_parts) > 0:
                ind = np.concatenate([np.asarray(part) for part in pred_parts])
        return ind, dep

    @staticmethod
    def _mode_payload_attack_val(
        *,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        attack: AttackLike | None,
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        del model
        assert data is not None and attack is not None
        dep = getattr(attack, "attacked_labels", None)
        if dep is None:
            dep = data.y_val
        return attack.attack_predictions, dep

    @staticmethod
    def _mode_payload_pre_sample(
        *,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        attack: AttackLike | None,
    ) -> tuple[MatrixLike | ArrayLike | None, MatrixLike | ArrayLike | None]:
        del model, attack
        assert data is not None
        dep = getattr(data, "y", getattr(data, "_y", None))
        ind = getattr(data, "X", getattr(data, "_X", None))
        if dep is None or ind is None:
            raise ValueError(
                "pre-sample mode requires data.X/data.y (or data._X/data._y) to be loaded",
            )
        return ind, dep

    @staticmethod
    def _normalize_existing_stage_results(
        results: dict[str, Any],
        stage_key: str,
    ) -> dict[str, Any]:
        existing_stage = results.get(stage_key)
        if isinstance(existing_stage, dict):
            return dict(existing_stage)
        if len(results) > 0 and all(not isinstance(v, dict) for v in results.values()):
            return dict(results)
        return {}

    @staticmethod
    def _resolve_attack_placeholders(
        kwargs: dict[str, Any],
        attack: AttackLike | None,
    ) -> None:
        if attack is None:
            return
        for key, value in kwargs.items():
            if value == "{attack}":
                kwargs[key] = getattr(
                    attack,
                    "attack",
                    getattr(attack, "_attack", None),
                )

    def _evaluate_stage_scorers(
        self,
        *,
        stage_results: dict[str, Any],
        runtime_stage_tokens: set[str],
        scorer_is_data_profile: bool,
        effective_mode: str,
        y_proba: MatrixLike | ArrayLike | None,
        dep: MatrixLike | ArrayLike | None,
        ind: MatrixLike | ArrayLike | None,
        data: "DataConfig | None",
        model: EstimatorLike | None,
        runtime_kwargs: dict[str, Any],
    ) -> None:
        for key, scorer in self.scorers.items():
            if not self._stage_matches(
                getattr(scorer, "stage", ""),
                runtime_stage_tokens,
            ):
                continue
            if stage_results.get(key) is not None:
                continue

            metric_input = self._resolve_metric_input(
                key=key,
                scorer=scorer,
                scorer_is_data_profile=scorer_is_data_profile,
                effective_mode=effective_mode,
                y_proba=y_proba,
                dep=dep,
                ind=ind,
                data=data,
                model=model,
            )
            value = scorer(
                dep=dep,
                ind=metric_input,
                **runtime_kwargs,
            )
            logger.debug(
                "Scorer '%s' raw output: %s (type: %s)",
                key,
                value,
                type(value),
            )
            if isinstance(value, (dict, pd.Series, pd.DataFrame)):
                flat_scores = _series_like_to_float_dict(value)
                for metric_name, metric_value in flat_scores.items():
                    stage_results[f"{key}_{metric_name}"] = metric_value
            else:
                stage_results[key] = _series_like_to_float_dict(value)["value"]

    def _resolve_metric_input(
        self,
        *,
        key: str,
        scorer: "ScorerConfig",
        scorer_is_data_profile: bool,
        effective_mode: str,
        y_proba: MatrixLike | ArrayLike | None,
        dep: MatrixLike | ArrayLike | None,
        ind: MatrixLike | ArrayLike | None,
        data: "DataConfig | None",
        model: EstimatorLike | None,
    ) -> MatrixLike | ArrayLike | None:
        metric_input: MatrixLike | ArrayLike | None = ind
        if scorer_is_data_profile and scorer.needs_proba is True:
            raise ValueError(
                f"Scorer '{key}' is configured as data-profile but requests probability outputs.",
            )
        if scorer.needs_proba is not True:
            return metric_input
        if effective_mode == "pre-sample":
            raise ValueError(
                f"Scorer '{key}' requires raw model outputs but pre-sample mode is reserved for full-dataset diagnostics.",
            )
        if y_proba is not None:
            metric_input = y_proba
        else:
            X_mode = self.resolve_mode_features(
                mode=effective_mode,
                data=data,
            )
            if X_mode is None or model is None:
                raise ValueError(
                    f"Scorer '{key}' requires raw model outputs from predict_proba; provide y_proba or pass model+data context",
                )
            metric_input = self.predict_proba_from_model(
                model=model,
                X=X_mode,
                y_true=dep,
                y_pred=ind,
            )

        metric_arr = np.asarray(to_numpy_if_torch(metric_input))
        if metric_arr.ndim not in (1, 2):
            raise ValueError(
                f"Scorer '{key}' expected 1D/2D raw output array, got shape {metric_arr.shape}. "
                "Check your model/scorer configuration.",
            )
        return metric_input

    def __call__(
        self,
        mode: str | None = None,
        data: "DataConfig | None" = None,
        model: EstimatorLike | None = None,
        attack: AttackLike | None = None,
        ind: MatrixLike | ArrayLike | None = None,
        dep: MatrixLike | ArrayLike | None = None,
        score_file: str | None = None,
        **kwargs: ScoreKwargValue,
    ) -> ScoreDict:
        """Execute staged scorer evaluation and return a normalized score payload.

        Args:
            mode: Requested scoring mode.
            data: Optional data runtime context.
            model: Optional model runtime context.
            attack: Optional attack runtime context.
            ind: Independent payload (features/predictions).
            dep: Dependent payload (labels/targets).
            score_file: Optional persisted score file path.
            **kwargs: Additional scoring/runtime kwargs.

        Returns:
            Canonical score payload for resolved stage.

        Raises:
            AssertionError: If no runtime payload can be resolved for scoring.
            ValueError: If mode/stage resolution or payload requirements are invalid.
            TypeError: If configured scorer entries are not callable.
            KeyError: If scoring mode token is unsupported.
        """
        results: dict[str, Any] = {}
        runtime_stage = kwargs.pop("stage", None)

        if (
            mode is None
            and runtime_stage is None
            and ind is None
            and dep is None
            and "y_pred" not in kwargs
            and "y_true" not in kwargs
        ):
            raise AssertionError("y_true must be provided if mode is None")

        effective_mode = self._resolve_runtime_mode(
            mode=mode,
            requested_stage=runtime_stage,
        )
        scorer_is_data_profile = self._is_data_profile_scorer()
        if effective_mode == "pre-sample" and not scorer_is_data_profile:
            raise ValueError(
                "pre-sample mode is reserved for data-profile scorers.",
            )

        if score_file is not None and Path(score_file).exists():
            results = self.load_scores(score_file)

        if not isinstance(results, dict):
            results = {}

        stage_key = self._resolve_stage_key(
            mode=mode,
            requested_stage=runtime_stage,
        )
        stage_results = self._normalize_existing_stage_results(results, stage_key)

        runtime_stage_tokens = self._runtime_stage_tokens(
            mode=effective_mode,
            stage=runtime_stage,
        )
        if not self._stage_matches(self.stage, runtime_stage_tokens):
            raise KeyError(
                "ScorerDictConfig stage filter did not match requested stage. "
                f"configured={self.stage}, runtime={sorted(runtime_stage_tokens)}",
            )

        ind, dep = self._resolve_ind_dep_from_kwargs(ind, dep, kwargs)

        if ind is not None:
            if dep is None:
                raise AssertionError(
                    "If y_pred is provided, y_true must also be provided.",
                )
        else:
            ind, dep = self._resolve_mode_payload(
                effective_mode=effective_mode,
                data=data,
                model=model,
                attack=attack,
            )
            if dep is None:
                raise AssertionError("y_true must be provided if mode is None")

        self._resolve_attack_placeholders(kwargs, attack)

        y_proba = kwargs.pop("y_proba", None)

        runtime_kwargs = {
            **kwargs,
            "data": data,
            "model": model,
            "attack": attack,
            "mode": effective_mode,
        }

        if not self.scorers:
            raise ValueError(
                "ScorerDictConfig must have at least one scorer defined; got empty scorers dict.",
            )

        self._evaluate_stage_scorers(
            stage_results=stage_results,
            runtime_stage_tokens=runtime_stage_tokens,
            scorer_is_data_profile=scorer_is_data_profile,
            effective_mode=effective_mode,
            y_proba=y_proba,
            dep=dep,
            ind=ind,
            data=data,
            model=model,
            runtime_kwargs=runtime_kwargs,
        )

        if not stage_results:
            raise KeyError(
                f"No scores found for requested stage '{stage_key}'.",
            )

        results[stage_key] = stage_results

        if score_file is not None:
            self.save_scores(results, score_file)
        if mode is not None or runtime_stage is not None:
            return ScoreDict.from_payload({stage_key: results[stage_key]})
        return ScoreDict.from_payload(results[stage_key])


def coerce_scorer_config(scorer_obj, *, default_factory=None):
    """Unified scorer coercion for DataConfig, ModelConfig, and ExperimentConfig.

    Converts any scorer spec into a :class:`ScorerDictConfig` (or ``None``).

    Args:
        scorer_obj: Raw scorer value from a config field.
        default_factory: Zero-argument callable returning the default scorer
            when ``scorer_obj`` is a default token such as ``"auto"``,
            ``"default"``, or ``"best"``. When ``None``, default tokens are
            treated as null and the function returns ``None``.
    """

    if is_null_config_value(scorer_obj):
        return None
    if is_default_config_value(scorer_obj, include_best=True):
        if default_factory is not None:
            return default_factory()
        return None
    # Specialized configs may provide ready-to-use scorer runtime objects
    # (e.g., custom scorer classes instantiated via load_class).
    if callable(scorer_obj):
        return scorer_obj
    if isinstance(scorer_obj, ScorerDictConfig):
        return scorer_obj
    if isinstance(scorer_obj, (list, ListConfig)):
        return ScorerDictConfig.merge(list(scorer_obj))
    scorer_obj = coerce_config(
        scorer_obj,
    )  # DictConfig->dict, BaseConfig->dict, YAML file->dict
    if isinstance(scorer_obj, str):
        scorer_obj = ScorerDictConfig.from_yaml(scorer_obj).to_dict()
    if isinstance(scorer_obj, dict):
        if "_target_" in scorer_obj:
            # Preserve concrete type info (e.g. _DataScorerMarker, _AttackProfileScorer)
            return instantiate(scorer_obj)
        if "scorers" in scorer_obj:
            try:
                return ScorerDictConfig(**scorer_obj)
            except TypeError:
                # Some structured task-aware scorer objects may be converted to
                # dicts without `_target_` (e.g. contain `classifier` + `scorers`).
                # In that case keep the scorer payload and drop task metadata.
                fallback = dict(scorer_obj)
                fallback.pop("classifier", None)
                if "group_scorers" in fallback:
                    try:
                        from ..plugins.fairlearn.score import FairlearnScorerDictConfig

                        return FairlearnScorerDictConfig(**fallback)
                    except Exception:
                        pass
                return ScorerDictConfig(scorers=fallback.get("scorers", {}))
        return ScorerDictConfig(scorers=scorer_obj)
    raise ValueError(f"Unsupported scorer config type: {type(scorer_obj)}")


def build_scorer(cfg: ScorerConfig):
    return cfg if isinstance(cfg, ScorerConfig) else ScorerConfig(**cfg)


def build_scorer_dict(cfg: ScorerDictConfig):
    return cfg if isinstance(cfg, ScorerDictConfig) else ScorerDictConfig(**cfg)


def default_weighted_classification_core_scorers(
    *,
    f1_key: str = "f1",
) -> dict[str, ScorerConfig]:
    """Return reusable weighted classification metrics without proba-only scorers."""
    return {
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
        f1_key: ScorerConfig(
            score_name=f1_key,
            score_function="sklearn.metrics.f1_score",
            score_params={"average": "weighted", "zero_division": 0},
        ),
    }


def default_regression_scorers() -> dict[str, ScorerConfig]:
    """Return reusable default regression metrics."""
    return {
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
    }


def _default_classification_scorers() -> dict[str, ScorerConfig]:
    return {
        **default_weighted_classification_core_scorers(),
        "roc_auc": ScorerConfig(
            score_name="roc_auc",
            score_function="sklearn.metrics.roc_auc_score",
            score_params={"average": "weighted", "multi_class": "ovr"},
            needs_labels=False,
            needs_proba=True,
            needs_logits=True,
        ),
        "log_loss": ScorerConfig(
            score_name="log_loss",
            score_function="sklearn.metrics.log_loss",
            needs_labels=False,
            needs_proba=True,
            needs_logits=True,
        ),
    }


def _default_regression_scorers() -> dict[str, ScorerConfig]:
    return default_regression_scorers()


def _default_pytorch_classification_scorers() -> dict[str, ScorerConfig]:
    return default_weighted_classification_core_scorers()


@dataclass(eq=False, kw_only=True)
class DefaultModelScorerDictConfig(TaskAwareScorerMixin, ScorerDictConfig):
    """Default model scorer family with optional task inheritance.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    classifier: Union[bool, str, None] = None
    scoring_type: str = "model"
    scorers: dict[str, ScorerConfig] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: scorers."},
    )

    def _build_default_scorers(self, classifier: bool) -> dict[str, ScorerConfig]:
        return (
            _default_classification_scorers()
            if classifier
            else _default_regression_scorers()
        )

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False, kw_only=True)
class DefaultClassifierScorerDictConfig(DefaultModelScorerDictConfig):
    """DefaultClassifierScorerDictConfig runtime class.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    classifier: Union[bool, str, None] = True


@dataclass(eq=False, kw_only=True)
class DefaultRegressorScorerDictConfig(DefaultModelScorerDictConfig):
    """DefaultRegressorScorerDictConfig runtime class.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    classifier: Union[bool, str, None] = False


@dataclass(eq=False, kw_only=True)
class DefaultPytorchScorerDictConfig(TaskAwareScorerMixin, ScorerDictConfig):
    """Default PyTorch scorer family with optional task inheritance.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    classifier: Union[bool, str, None] = None
    scoring_type: str = "model"
    scorers: dict[str, ScorerConfig] = field(
        default_factory=dict,
        metadata={"help": "Configuration field: scorers."},
    )

    def _build_default_scorers(self, classifier: bool) -> dict[str, ScorerConfig]:
        return (
            _default_pytorch_classification_scorers()
            if classifier
            else _default_regression_scorers()
        )

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False, kw_only=True)
class DefaultPytorchClassifierScorerDictConfig(DefaultPytorchScorerDictConfig):
    """Default classifier scorers for PyTorch models.

    PyTorch model wrappers often expose logits but not ``predict_proba``. This
    default avoids probability-required metrics so automatic scoring works out
    of the box.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    classifier: Union[bool, str, None] = True


@dataclass(eq=False, kw_only=True)
class DefaultPytorchRegressorScorerDictConfig(DefaultPytorchScorerDictConfig):
    """Default regressor scorers for PyTorch models.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    classifier: Union[bool, str, None] = False


safe_store(
    group="score",
    name="classification",
    node={
        "_target_": "deckard.score.base.DefaultModelScorerDictConfig",
        "classifier": True,
    },
)
safe_store(
    group="score",
    name="regression",
    node={
        "_target_": "deckard.score.base.DefaultModelScorerDictConfig",
        "classifier": False,
    },
)
safe_store(
    group="score",
    name="pytorch_classification",
    node={
        "_target_": "deckard.score.base.DefaultPytorchScorerDictConfig",
        "classifier": True,
    },
)
safe_store(
    group="score",
    name="pytorch_regression",
    node={
        "_target_": "deckard.score.base.DefaultPytorchScorerDictConfig",
        "classifier": False,
    },
)


__all__ = [
    "safe_store",
    "SUPPORTED_SCORING_STAGES",
    "SUPPORTED_DATA_SCORE_MODES",
    "SUPPORTED_MODEL_SCORE_MODES",
    "SUPPORTED_EXPERIMENT_DEFENSE_SCORING_STAGES",
    "SUPPORTED_ATTACK_SCORE_MODES",
    "SUPPORTED_EXPERIMENT_SCORE_MODES",
    "SUPPORTED_DETECTOR_SCORE_MODES",
    "SUPPORTED_PIPELINE_SCORE_MODES",
    "_DataScorerMarker",
    "_AttackProfileScorer",
    "TaskAwareScorerMixin",
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultModelScorerDictConfig",
    "DefaultClassifierScorerDictConfig",
    "DefaultRegressorScorerDictConfig",
    "DefaultPytorchScorerDictConfig",
    "DefaultPytorchClassifierScorerDictConfig",
    "DefaultPytorchRegressorScorerDictConfig",
    "build_scorer",
    "build_scorer_dict",
]
