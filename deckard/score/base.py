"""Core scoring primitives and default scorer profiles."""

from dataclasses import dataclass, field
import inspect
import logging
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Union, cast

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig

from ..utils import (
    ConfigBase,
    coerce_config,
    is_default_config_value,
    is_null_config_value,
    resolve_class,
    safe_store,
    merge_list_of_dicts,
    load_class,
)
if TYPE_CHECKING:
    from ..data import DataConfig

from .pytorch import to_numpy_if_torch

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


def _normalize_classifier_flag(classifier: Union[bool, str, None]) -> Union[bool, None]:
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

    def _initialize_task_aware_scorers(self, *, default: Union[bool, None] = None) -> None:
        self._normalize_classifier()
        if getattr(self, "scorers", None):
            return
        classifier = self.resolve_classifier(default=default)
        self.scorers = self._build_default_scorers(classifier=classifier)


def _resolve_yt_yp(
    mode: Union[Literal["test", "train", "attack", "val", "attack-val"], None],
    data: "DataConfig | None",
    model: Any,
    attack: Any,
    y_pred: Any,
    y_true: Any,
) -> tuple[Any, Any]:
    """Resolve y_true and y_pred from mode + context when not explicitly provided.

    This mirrors the resolution logic inside ``ScorerDictConfig.__call__`` so that
    mixin overrides (e.g. ``_FairnessScorerMixin``) can access the resolved values
    after delegating to ``super().__call__()``.
    """
    if y_pred is not None:
        return y_true, y_pred
    if mode == "test":
        if data is not None:
            y_true = getattr(data, "y_test", y_true)
        if model is not None:
            y_pred = getattr(model, "test_predictions", None) or getattr(model, "predictions", None)
    elif mode == "train":
        if data is not None:
            y_true = getattr(data, "y_train", y_true)
        if model is not None:
            y_pred = getattr(model, "training_predictions", None)
    elif mode == "attack":
        if data is not None and attack is not None:
            y_test = np.asarray(getattr(data, "y_test", y_true))
            attack_size = getattr(attack, "attack_size", None)
            y_true = y_test[:attack_size] if attack_size is not None else y_test
        if attack is not None:
            y_pred = getattr(attack, "attack_predictions", None)
    elif mode == "val":
        if data is not None:
            y_true = getattr(data, "y_val", y_true)
        if model is not None:
            y_pred = getattr(model, "val_predictions", None)
    elif mode == "attack-val":
        if data is not None:
            y_true = getattr(data, "y_val", y_true)
        if attack is not None:
            y_pred = getattr(attack, "attack_predictions", None)
    return y_true, y_pred


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

    def __call__(
        self,
        y_true: Any,
        y_pred: Any,
        swap: bool = False,
        **kwargs: Any,
    ) -> MetricResult:
        if swap:
            y_true, y_pred = y_pred, y_true
        y_true = to_numpy_if_torch(y_true)
        y_pred = to_numpy_if_torch(y_pred)
        y_pred = self._normalize_predictions_for_metric(
            y_true=y_true,
            y_pred=y_pred,
        )
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
            params = {k: v for k, v in params.items() if k in accepted}
        return cast(MetricResult, score_function(y_true, y_pred, **params))


@dataclass(eq=False)
class ScorerDictConfig(ConfigBase):
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
                    greater_is_better=bool(scorer_data.pop(
                        "greater_is_better",
                        True,
                    )),
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
                    greater_is_better=bool(scorer_data.pop(
                        "greater_is_better",
                        True,
                    )),
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

    def get_callables(self):
        return {key: scorer for key, scorer in self.scorers.items()}

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
        predict_proba = getattr(estimator, "predict_proba")
        if not callable(predict_proba):
            raise ValueError(
                "Probability-required scorer configured but predict_proba is not callable",
            )

        try:
            return predict_proba(X)
        except TypeError:
            return predict_proba(np.asarray(X, dtype=float))

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
        data: "DataConfig | None" = None,
        model: Any = None,
        attack: Any = None,
        y_pred=None,
        y_true=None,
        score_file=None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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
                assert data is not None
                y_true = data.y_test
                y_pred = getattr(model, "test_predictions", None)
                if y_pred is None:
                    y_pred = getattr(model, "predictions", None)
            elif mode == "train":
                assert data is not None and model is not None
                y_true = data.y_train
                y_pred = model.training_predictions
            elif mode == "attack":
                assert data is not None and attack is not None
                y_test = getattr(data, "y_test", None)
                if y_test is None:
                    raise ValueError("attack mode requires data.y_test")
                y_true = y_test[: attack.attack_size]
                y_pred = attack.attack_predictions
            elif mode == "val":
                assert data is not None and model is not None
                y_true = data.y_val
                y_pred = model.val_predictions
            elif mode == "attack-val":
                assert data is not None and attack is not None
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
    scorer_obj = coerce_config(scorer_obj)  # DictConfig->dict, ConfigBase->dict, YAML file->dict
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
                        from .fairness import FairlearnScoreDictConfig

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


@dataclass(eq=False)
class DefaultModelScoreConfig(_TaskAwareScorerMixin, ScorerDictConfig):
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


@dataclass(eq=False)
class DefaultClassifierConfig(DefaultModelScoreConfig):
    classifier: Union[bool, str, None] = True


@dataclass(eq=False)
class DefaultRegressorConfig(DefaultModelScoreConfig):
    classifier: Union[bool, str, None] = False


@dataclass(eq=False)
class DefaultPytorchScoreConfig(_TaskAwareScorerMixin, ScorerDictConfig):
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


@dataclass(eq=False)
class DefaultPytorchClassifierConfig(DefaultPytorchScoreConfig):
    """Default classifier scorers for PyTorch models.

    PyTorch model wrappers often expose logits but not ``predict_proba``. This
    default avoids probability-required metrics so automatic scoring works out
    of the box.
    """

    classifier: Union[bool, str, None] = True


@dataclass(eq=False)
class DefaultPytorchRegressorConfig(DefaultPytorchScoreConfig):
    """Default regressor scorers for PyTorch models."""

    classifier: Union[bool, str, None] = False


safe_store(
    group="score",
    name="classification",
    node={"_target_": "deckard.score.base.DefaultModelScoreConfig", "classifier": True},
)
safe_store(
    group="score",
    name="regression",
    node={"_target_": "deckard.score.base.DefaultModelScoreConfig", "classifier": False},
)
safe_store(
    group="score",
    name="pytorch_classification",
    node={"_target_": "deckard.score.base.DefaultPytorchScoreConfig", "classifier": True},
)
safe_store(
    group="score",
    name="pytorch_regression",
    node={"_target_": "deckard.score.base.DefaultPytorchScoreConfig", "classifier": False},
)


__all__ = [
    "safe_store",
    "_DataScorerMarker",
    "_AttackProfileScorer",
    "_TaskAwareScorerMixin",
    "_resolve_yt_yp",
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultModelScoreConfig",
    "DefaultClassifierConfig",
    "DefaultRegressorConfig",
    "DefaultPytorchScoreConfig",
    "DefaultPytorchClassifierConfig",
    "DefaultPytorchRegressorConfig",
    "build_scorer",
    "build_scorer_dict",
]
