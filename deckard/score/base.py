"""Core scoring primitives and default scorer profiles."""

from dataclasses import dataclass, field
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Union

from hydra.core.config_store import ConfigStore

from ..utils import ConfigBase, resolve_class

logger = logging.getLogger(__name__)


def safe_store(group: str, name: str, node):
    """Register a config node while tolerating duplicate import-time stores."""
    cs = ConfigStore.instance()
    try:
        cs.store(group=group, name=name, node=node)
    except Exception:
        # Re-imports can attempt duplicate registrations in some test contexts.
        pass


@dataclass
class ScorerConfig:
    """Atomic scorer configuration."""

    score_name: str
    score_function: Union[str, Callable]
    score_params: Dict[str, Any] = field(default_factory=dict)
    greater_is_better: bool = True
    needs_proba: bool = False

    def __post_init__(self):
        if isinstance(self.score_function, str):
            self.score_function = resolve_class(self.score_function)
        if not callable(self.score_function):
            raise TypeError("score_function must be callable or import path string")
        if self.score_params is None:
            self.score_params = {}

    def __call__(self, y_true, y_pred, swap: bool = False, **kwargs):
        if swap:
            y_true, y_pred = y_pred, y_true
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


@dataclass
class ScorerDictConfig(ConfigBase):
    """Container of named ScorerConfig instances."""

    scorers: Dict[str, Union[ScorerConfig, Dict[str, Any]]] = field(
        default_factory=dict
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
                    greater_is_better=scorer_data.pop("greater_is_better", True),
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

    def __getitem__(self, key):
        return self.scorers[key]

    def get_callables(self):
        return {key: scorer for key, scorer in self.scorers.items()}

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

        for key, scorer in self.scorers.items():
            scored_key = key
            if mode == "train":
                scored_key = f"training_{key}"
            elif mode == "attack":
                scored_key = f"attack_{key}"
            if results.get(scored_key) is None:
                results[scored_key] = scorer(y_true=y_true, y_pred=y_pred, **kwargs)

        if score_file is not None:
            self.save_scores(results, score_file)
        return results


def build_scorer(cfg: ScorerConfig):
    return cfg if isinstance(cfg, ScorerConfig) else ScorerConfig(**cfg)


def build_scorer_dict(cfg: ScorerDictConfig):
    return cfg if isinstance(cfg, ScorerDictConfig) else ScorerDictConfig(**cfg)


@dataclass
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
            ),
            "log_loss": ScorerConfig(
                score_name="log_loss",
                score_function="sklearn.metrics.log_loss",
            ),
        },
    )


@dataclass
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


safe_store(group="score", name="classification", node=DefaultClassifierConfig)
safe_store(group="score", name="regression", node=DefaultRegressorConfig)


__all__ = [
    "safe_store",
    "ScorerConfig",
    "ScorerDictConfig",
    "DefaultClassifierConfig",
    "DefaultRegressorConfig",
    "build_scorer",
    "build_scorer_dict",
]
