from __future__ import annotations

"""Core scoring primitives and default scorer profiles."""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from ..frameworks import FrameworkDataScorer, ScorerContractMixin
from ..frameworks.core import ArrayLike, MatrixLike
from ..frameworks.pytorch.score import to_numpy_if_torch
from ..utils import (
    ConfigBase,
    coerce_config,
    is_default_config_value,
    is_null_config_value,
    load_class,
    merge_list_of_dicts,
    resolve_class,
    safe_store,
)
from ._runtime import series_like_to_float_dict as _series_like_to_float_dict

if TYPE_CHECKING:
    from ..data import DataConfig

logger = logging.getLogger(__name__)

MetricScalar = Union[float, int, np.floating, np.integer]
MetricResult = Union[MetricScalar, np.ndarray]
ScoreFunction = Callable[..., MetricResult]


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


@dataclass(eq=True)
class _ScorerMixin:
    """Base callable scorer handler used by runtime scorer context resolution.

    Initialization parameters
    -------------------------
    runtime : ScorerDictConfig
        Runtime config object owned by ``ScorerDictConfig.__call__``. Mixins should
        treat this as the source of mutable runtime state (scorers, cached results, etc).

    Runtime parameters
    -------------------
    The mixin forwards attribute access to runtime to enable transparent delegation
    of scorer configuration to the underlying runtime instance.
    """

    runtime: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runtime, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "runtime":
            object.__setattr__(self, name, value)
            return
        runtime = object.__getattribute__(self, "runtime")
        if runtime is None:
            object.__setattr__(self, name, value)
            return
        setattr(runtime, name, value)

    def __call__(
        self,
        *,
        data: Any = None,
        model: Any = None,
        attack: Any = None,
        mode: str = "test",
    ) -> dict:
        """Execute one scorer handler.

        Parameters
        ----------
        data : Any
            Data runtime containing train/test/val splits.
        model : Any
            User model object or model config supplied to scorer chain.
        attack : Any
            Instantiated attack object (optional).
        mode : str
            Scoring mode (test, train, attack, val, attack-val, pre-sample).

        Returns
        -------
        dict
            Score dictionary from scorer execution.
        """
        raise NotImplementedError(
            "Scorer mixins must implement __call__",
        )


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
    """

    mixin_type: Any
    scoring_type: str
    scoring_subtype: Union[str, None] = None
    excluded_subtypes: tuple[str, ...] = field(default_factory=tuple)
    init_params: dict[str, Any] = field(default_factory=dict)

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
        """Return mixin tuple for matching scoring family/subtype."""
        _ = (runtime, default_mixins)
        if not self._matches(
            scoring_type=scoring_type,
            scoring_subtype=scoring_subtype,
        ):
            return tuple()
        mixin = self._resolve_mixin_type()
        return (mixin,)

    def resolve_scorer_handler(
        self,
        runtime: "ScorerDictConfig",
        *,
        scoring_type: str,
        scoring_subtype: Union[str, None],
        default_handler: Any,
        default_mixins: tuple[type, ...],
    ) -> Any:
        """Return callable runtime handler for matching scoring family/subtype."""
        _ = (default_handler, default_mixins)
        if not self._matches(
            scoring_type=scoring_type,
            scoring_subtype=scoring_subtype,
        ):
            return None
        return lambda *args, **kwargs: self(runtime, *args, **kwargs)

    def __call__(self, runtime: "ScorerDictConfig", *args, **kwargs) -> dict:
        """Delegate runtime scorer execution to configured mixin handler.

        Parameters
        ----------
        runtime : ScorerDictConfig
            Runtime config instance currently orchestrating scoring.
        *args : Any
            Positional runtime args forwarded to mixin ``__call__``.
        **kwargs : Any
            Keyword runtime args forwarded to mixin ``__call__``.
        """
        mixin = self._resolve_mixin_type()
        handler = mixin(runtime)
        return handler(*args, **kwargs)


def _normalize_classifier_flag(
    classifier: Union[bool, str, None],
) -> Union[bool, None]:
    """Normalize classifier/regressor aliases to ``True`` / ``False`` / ``None``."""
    if classifier in ["classifier", True]:
        return True
    if classifier in ["regressor", False]:
        return False
    return None


class _TaskAwareScorerMixin:
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
    """

    classifier: Union[bool, str, None] = None

    def _normalize_classifier(self) -> None:
        self.classifier = _normalize_classifier_flag(getattr(self, "classifier", None))

    def resolve_classifier(
        self,
        *,
        data: "DataConfig | None" = None,
        model: Any = None,
        attack: Any = None,
        default: Union[bool, None] = None,
    ) -> bool:
        """Resolve the effective task type for this scorer config.

        Precedence is:
        1. explicit ``self.classifier``
        2. attack-derived runtime context
        3. ``model.classifier``
        4. ``data.classifier``
        5. explicit ``default``
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
    """Atomic scorer configuration."""

    score_name: str
    score_function: Any
    score_params: dict[str, Any] = field(default_factory=dict)
    greater_is_better: bool = True
    needs_proba: bool = False

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

    def _validate_probability_input(self, y_true, y_pred):
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
                f"Probability scorer '{self.score_name}' requires numeric probabilities; got dtype {y_pred_arr.dtype}",
            )
        if np.nanmin(y_pred_arr) < -1e-12 or np.nanmax(y_pred_arr) > 1.0 + 1e-12:
            raise ValueError(
                f"Probability scorer '{self.score_name}' requires values in [0, 1]; got min={np.nanmin(y_pred_arr)} max={np.nanmax(y_pred_arr)}",
            )

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
            "group_mean_prediction_difference",
            "group_mae_difference",
            "group_mse_difference",
        }
        is_label_metric = (
            metric_name in label_metrics or self.score_name in label_metrics
        )

        if self.needs_proba:
            ind_arr = np.asarray(to_numpy_if_torch(ind))
            if ind_arr.ndim == 2 and (
                np.nanmin(ind_arr) < 0.0 or np.nanmax(ind_arr) > 1.0
            ):
                exp_logits = np.exp(ind_arr - np.max(ind_arr, axis=1, keepdims=True))
                ind_arr = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            self._validate_probability_input(y_true=dep, y_pred=ind_arr)
            if ind_arr.ndim == 2 and metric_name == "roc_auc_score":
                if ind_arr.shape[1] == 1:
                    return ind_arr.reshape(-1)
                if ind_arr.shape[1] == 2:
                    return ind_arr[:, 1]
            return ind_arr

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
        dep: Any = None,
        ind: Any = None,
        swap: bool = False,
        **kwargs: Any,
    ) -> MetricResult:
        """Execute one scorer using generic dependent/independent payload names."""
        if dep is None and "y_true" in kwargs:
            dep = kwargs.pop("y_true")
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
class ScorerDictConfig(ScorerContractMixin, ConfigBase, FrameworkDataScorer):
    """Container of named ScorerConfig instances."""

    scorers: dict[str, ScorerConfig] = field(
        default_factory=dict,
    )

    def __post_init__(self):
        normalized = {}
        for key, value in self.scorers.items():
            if isinstance(value, ScorerConfig):
                scorer = value
            elif isinstance(value, dict):
                scorer_data = dict(value)
                raw_score_name = scorer_data.pop("score_name", key)
                raw_score_params = scorer_data.pop("score_params", {})
                if not isinstance(raw_score_params, dict):
                    raise TypeError(
                        f"score_params for '{key}' must be a dictionary, got {type(raw_score_params)}",
                    )
                scorer = ScorerConfig(
                    score_name=str(raw_score_name),
                    score_function=scorer_data.pop("score_function"),
                    score_params=dict(raw_score_params),
                    greater_is_better=bool(
                        scorer_data.pop(
                            "greater_is_better",
                            True,
                        ),
                    ),
                    needs_proba=bool(scorer_data.pop("needs_proba", False)),
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
                if not isinstance(raw_score_params, dict):
                    raise TypeError(
                        f"score_params for '{key}' must be a dictionary, got {type(raw_score_params)}",
                    )
                scorer = ScorerConfig(
                    score_name=str(raw_score_name),
                    score_function=scorer_data.pop("score_function"),
                    score_params=dict(raw_score_params),
                    greater_is_better=bool(
                        scorer_data.pop(
                            "greater_is_better",
                            True,
                        ),
                    ),
                    needs_proba=bool(scorer_data.pop("needs_proba", False)),
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

    @property
    def configured_scorers(self) -> dict[str, ScorerConfig]:
        """Public accessor for configured scorer definitions."""
        return self.scorers

    @configured_scorers.setter
    def configured_scorers(self, value: dict[str, ScorerConfig] | None) -> None:
        """Set configured scorer definitions."""
        self.scorers = value or {}

    def get_callables(self):
        return {key: scorer for key, scorer in self.scorers.items()}

    def score(
        self,
        ind: MatrixLike,
        dep: ArrayLike,
        *args: Any,
        data: "DataConfig | None" = None,
        model: Any = None,
        attack: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
    def merge(cls, items):
        """Merge a list of scorer specs into a single ScorerDictConfig.

        Each element of *items* may be a :class:`ScorerDictConfig`, a dict
        with a ``scorers`` key, or a bare scorers dict (name -> scorer spec).
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
    def resolve_mode_features(mode, data):
        if data is None:
            return None
        if mode == "train":
            return getattr(data, "X_train", None)
        if mode == "test":
            return getattr(data, "X_test", None)
        if mode in {"val", "attack-val"}:
            return getattr(data, "X_val", None)
        if mode == "pre-sample":
            return getattr(data, "X", getattr(data, "_X", None))
        return None

    @staticmethod
    def is_classification_labels(y):
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
    def predict_proba_from_model(model, X, y_true=None, y_pred=None):
        """
        For torch models, use the model's raw output (logits) as the probability input for normalization if predict_proba is not available.
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

    def __call__(
        self,
        mode: Literal[
            "test",
            "train",
            "attack",
            "val",
            "attack-val",
            "pre-sample",
            None,
        ] = "test",
        data: "DataConfig | None" = None,
        model: Any = None,
        attack: Any = None,
        ind=None,
        dep=None,
        score_file=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        results = {}
        if score_file is not None and Path(score_file).exists():
            results = self.load_scores(score_file)

        if ind is None and "y_pred" in kwargs:
            ind = kwargs.pop("y_pred")
        if dep is None and "y_true" in kwargs:
            dep = kwargs.pop("y_true")

        if ind is not None:
            if dep is None:
                raise AssertionError(
                    "If y_pred is provided, y_true must also be provided.",
                )
        else:
            if mode == "test":
                assert data is not None
                dep = data.y_test
                ind = getattr(model, "test_predictions", None)
                if ind is None:
                    ind = getattr(model, "predictions", None)
            elif mode == "train":
                assert data is not None and model is not None
                dep = data.y_train
                ind = model.training_predictions
            elif mode == "attack":
                assert data is not None and attack is not None
                dep = getattr(attack, "attacked_labels", None)
                if dep is None:
                    y_test = getattr(data, "y_test", None)
                    if y_test is None:
                        raise ValueError(
                            "attack mode requires attack.attacked_labels or data.y_test",
                        )
                    dep = y_test[: attack.attack_size]
                ind = attack.attack_predictions
            elif mode == "val":
                assert data is not None and model is not None
                dep = data.y_val
                ind = model.val_predictions
            elif mode == "attack-val":
                assert data is not None and attack is not None
                dep = getattr(attack, "attacked_labels", None)
                if dep is None:
                    dep = data.y_val
                ind = attack.attack_predictions
            elif mode == "pre-sample":
                assert data is not None
                dep = getattr(data, "y", getattr(data, "_y", None))
                ind = getattr(data, "X", getattr(data, "_X", None))
                if dep is None or ind is None:
                    raise ValueError(
                        "pre-sample mode requires data.X/data.y (or data._X/data._y) to be loaded",
                    )
            elif dep is None:
                raise AssertionError("y_true must be provided if mode is None")

        if attack is not None:
            for key, value in kwargs.items():
                if value == "{attack}":
                    kwargs[key] = getattr(
                        attack,
                        "attack",
                        getattr(attack, "_attack", None),
                    )

        y_proba = kwargs.pop("y_proba", None)

        runtime_kwargs = {
            **kwargs,
            "data": data,
            "model": model,
            "attack": attack,
            "mode": mode,
        }

        if not self.scorers:
            raise ValueError(
                "ScorerDictConfig must have at least one scorer defined; got empty scorers dict.",
            )

        for key, scorer in self.scorers.items():
            scored_key = key
            if mode == "train":
                scored_key = f"training_{key}"
            elif mode == "attack":
                scored_key = f"attack_{key}"
            if results.get(scored_key) is None:
                metric_input = ind
                if scorer.needs_proba:
                    if mode == "pre-sample":
                        raise ValueError(
                            f"Scorer '{key}' requires probabilities but pre-sample mode is reserved for full-dataset diagnostics.",
                        )
                    if y_proba is not None:
                        metric_input = y_proba
                    else:
                        X_mode = self.resolve_mode_features(
                            mode=mode,
                            data=data,
                        )
                        if X_mode is not None and model is not None:
                            metric_input = self.predict_proba_from_model(
                                model=model,
                                X=X_mode,
                                y_true=dep,
                                y_pred=ind,
                            )
                        else:
                            raise ValueError(
                                f"Scorer '{key}' requires probabilities from predict_proba; provide y_proba or pass model+data context",
                            )
                    # Final check: ensure metric_input is valid
                    metric_arr = np.asarray(to_numpy_if_torch(metric_input))
                    if metric_arr.ndim not in (1, 2):
                        raise ValueError(
                            f"Scorer '{key}' expected 1D/2D probability array, got shape {metric_arr.shape}. "
                            f"Check your model/scorer configuration.",
                        )
                # Debug print: show raw output from each scorer
                value = scorer(
                    dep=dep,
                    ind=metric_input,
                    **runtime_kwargs,
                )
                logger.debug(
                    f"Scorer '{scored_key}' raw output: {value} (type: {type(value)})",
                )
                if isinstance(value, (dict, pd.Series, pd.DataFrame)):
                    flat_scores = _series_like_to_float_dict(value)
                    for k, v in flat_scores.items():
                        # Use informative keys: f"{scored_key}_{k}"
                        results[f"{scored_key}_{k}"] = v
                else:
                    results[scored_key] = value

        if score_file is not None:
            self.save_scores(results, score_file)
        if not results:
            raise ValueError(
                "ScorerDictConfig.__call__ did not produce any results; ensure at least one scorer is defined and executed.",
            )
        return results


def coerce_scorer_config(scorer_obj, *, default_factory=None):
    """Unified scorer coercion for DataConfig, ModelConfig, and ExperimentConfig.

    Converts any scorer spec into a :class:`ScorerDictConfig` (or ``None``).

    Parameters
    ----------
    scorer_obj:
        The raw scorer value from a config field.
    default_factory:
        A zero-argument callable that returns the default scorer when
        *scorer_obj* is a default token (``"auto"``, ``"default"``,
        ``"best"``).  If ``None``, default tokens are treated as null
        (returns ``None``).
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
    )  # DictConfig->dict, ConfigBase->dict, YAML file->dict
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
                        from ..plugins.fairlearn.score import FairlearnScoreDictConfig

                        return FairlearnScoreDictConfig(**fallback)
                    except Exception:
                        pass
                return ScorerDictConfig(scorers=fallback.get("scorers", {}))
        return ScorerDictConfig(scorers=scorer_obj)
    raise ValueError(f"Unsupported scorer config type: {type(scorer_obj)}")


def build_scorer(cfg: ScorerConfig):
    return cfg if isinstance(cfg, ScorerConfig) else ScorerConfig(**cfg)


def build_scorer_dict(cfg: ScorerDictConfig):
    return cfg if isinstance(cfg, ScorerDictConfig) else ScorerDictConfig(**cfg)


def _default_classification_scorers() -> dict[str, ScorerConfig]:
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
    }


def _default_regression_scorers() -> dict[str, ScorerConfig]:
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


def _default_pytorch_classification_scorers() -> dict[str, ScorerConfig]:
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
        "f1": ScorerConfig(
            score_name="f1",
            score_function="sklearn.metrics.f1_score",
            score_params={"average": "weighted", "zero_division": 0},
        ),
    }


@dataclass(eq=False, kw_only=True)
class DefaultModelScorerConfig(_TaskAwareScorerMixin, ScorerDictConfig):
    """Default model scorer family with optional task inheritance."""

    classifier: Union[bool, str, None] = None
    scorers: dict[str, ScorerConfig] = field(default_factory=dict)

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
class DefaultClassifierConfig(DefaultModelScorerConfig):
    classifier: Union[bool, str, None] = True


@dataclass(eq=False, kw_only=True)
class DefaultRegressorConfig(DefaultModelScorerConfig):
    classifier: Union[bool, str, None] = False


@dataclass(eq=False, kw_only=True)
class DefaultPytorchScorerConfig(_TaskAwareScorerMixin, ScorerDictConfig):
    """Default PyTorch scorer family with optional task inheritance."""

    classifier: Union[bool, str, None] = None
    scorers: dict[str, ScorerConfig] = field(default_factory=dict)

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
class DefaultPytorchClassifierConfig(DefaultPytorchScorerConfig):
    """Default classifier scorers for PyTorch models.

    PyTorch model wrappers often expose logits but not ``predict_proba``. This
    default avoids probability-required metrics so automatic scoring works out
    of the box.
    """

    classifier: Union[bool, str, None] = True


@dataclass(eq=False, kw_only=True)
class DefaultPytorchRegressorConfig(DefaultPytorchScorerConfig):
    """Default regressor scorers for PyTorch models."""

    classifier: Union[bool, str, None] = False


safe_store(
    group="score",
    name="classification",
    node={
        "_target_": "deckard.score.base.DefaultModelScorerConfig",
        "classifier": True,
    },
)
safe_store(
    group="score",
    name="regression",
    node={
        "_target_": "deckard.score.base.DefaultModelScorerConfig",
        "classifier": False,
    },
)
safe_store(
    group="score",
    name="pytorch_classification",
    node={
        "_target_": "deckard.score.base.DefaultPytorchScorerConfig",
        "classifier": True,
    },
)
safe_store(
    group="score",
    name="pytorch_regression",
    node={
        "_target_": "deckard.score.base.DefaultPytorchScorerConfig",
        "classifier": False,
    },
)


__all__ = [
    "safe_store",
    "_DataScorerMarker",
    "_AttackProfileScorer",
    "_TaskAwareScorerMixin",
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultModelScorerConfig",
    "DefaultClassifierConfig",
    "DefaultRegressorConfig",
    "DefaultPytorchScorerConfig",
    "DefaultPytorchClassifierConfig",
    "DefaultPytorchRegressorConfig",
    "build_scorer",
    "build_scorer_dict",
]
