"""Attack-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
import time
from typing import Any, Literal, Union

from sklearn.metrics import accuracy_score

from ..utils import ConfigBase, round_scores
from .base import (
    ScorerConfig,
    ScorerDictConfig,
    _AttackProfileScorer,
    _TaskAwareScorerMixin,
    coerce_scorer_config,
    safe_store,
)

__all__ = [
    "evasion_success_score",
    "DefaultEvasionScoreConfig",
    "DefaultEvasionAttackScorerConfig",
    "DefaultEvasionRegressionAttackScorerConfig",
    "DefaultMembershipInferenceScoreConfig",
    "DefaultMembershipInferenceAttackScorerConfig",
    "DefaultAttributeInferenceScoreConfig",
    "DefaultAttributeInferenceAttackScorerConfig",
    "DefaultAttributeInferenceRegressionAttackScorerConfig",
    "AttackScorerConfig",
    "FairlearnAttackScorerConfig",
]


def evasion_success_score(
    y_true: Any,
    y_pred: Any,
    ben_pred_labels: Any = None,
    **kwargs: Any,
) -> float:
    """Compute evasion success as one minus benign/adversarial agreement."""
    if ben_pred_labels is None:
        raise ValueError(
            "ben_pred_labels are required for evasion_success scoring",
        )
    return float(1 - accuracy_score(ben_pred_labels, y_pred))


@dataclass(eq=False)
class DefaultEvasionScoreConfig(
    _TaskAwareScorerMixin, _AttackProfileScorer, ScorerDictConfig
):
    """Default evasion scorer family with optional task selection."""

    _profile_attr = "evasion"
    classifier: Union[bool, str] = True
    scorers: dict[str, ScorerConfig] = field(default_factory=dict)

    def _build_default_scorers(self, classifier: bool) -> dict[str, ScorerConfig]:
        if classifier:
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
                "f1-score": ScorerConfig(
                    score_name="f1-score",
                    score_function="sklearn.metrics.f1_score",
                    score_params={"average": "weighted", "zero_division": 0},
                ),
                "success": ScorerConfig(
                    score_name="success",
                    score_function="deckard.score.attack.evasion_success_score",
                ),
            }
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

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False)
class DefaultEvasionAttackScorerConfig(DefaultEvasionScoreConfig):
    """Default scorer set for evasion attack evaluation."""

    classifier: Union[bool, str] = True


@dataclass(eq=False)
class DefaultEvasionRegressionAttackScorerConfig(DefaultEvasionScoreConfig):
    """Default scorer set for evasion attacks against regression models."""

    _profile_attr = "evasion_regression"
    classifier: Union[bool, str] = False


@dataclass(eq=False)
class DefaultMembershipInferenceScoreConfig(
    _TaskAwareScorerMixin, _AttackProfileScorer, ScorerDictConfig
):
    """Default membership-inference scorer family.

    Membership inference is always treated as a classification-style scoring
    problem, so the default is fixed to ``classifier=True``.
    """

    _profile_attr = "membership_inference"
    classifier: Union[bool, str] = True
    scorers: dict[str, ScorerConfig] = field(default_factory=dict)

    def _build_default_scorers(self, classifier: bool) -> dict[str, ScorerConfig]:
        _ = classifier
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

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False)
class DefaultMembershipInferenceAttackScorerConfig(
    DefaultMembershipInferenceScoreConfig
):
    """Default scorer set for membership inference attack evaluation."""


@dataclass(eq=False)
class DefaultAttributeInferenceScoreConfig(
    _TaskAwareScorerMixin, _AttackProfileScorer, ScorerDictConfig
):
    """Default attribute-inference scorer family with optional task selection."""

    _profile_attr = "attribute_inference"
    classifier: Union[bool, str] = True
    scorers: dict[str, ScorerConfig] = field(default_factory=dict)

    def _build_default_scorers(self, classifier: bool) -> dict[str, ScorerConfig]:
        if classifier:
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

    def __post_init__(self):
        self._initialize_task_aware_scorers(default=True)
        super().__post_init__()


@dataclass(eq=False)
class DefaultAttributeInferenceAttackScorerConfig(
    DefaultAttributeInferenceScoreConfig
):
    """Default scorer set for categorical attribute inference evaluation."""

    classifier: Union[bool, str] = True


@dataclass(eq=False)
class DefaultAttributeInferenceRegressionAttackScorerConfig(
    DefaultAttributeInferenceScoreConfig
):
    """Default scorer set for continuous attribute inference evaluation."""

    _profile_attr = "attribute_inference_regression"
    classifier: Union[bool, str] = False


@dataclass(eq=False)
class AttackScorerConfig(ConfigBase):
    """Owns all attack scoring logic and profile-specific scorer configs."""

    evasion: Union[ScorerDictConfig, dict, None] = None
    evasion_regression: Union[ScorerDictConfig, dict, None] = None
    membership_inference: Union[ScorerDictConfig, dict, None] = None
    attribute_inference: Union[ScorerDictConfig, dict, None] = None
    attribute_inference_regression: Union[ScorerDictConfig, dict, None] = None

    def __post_init__(self):
        self.evasion = self._coerce_profile(
            self.evasion,
            DefaultEvasionAttackScorerConfig,
        )
        self.evasion_regression = self._coerce_profile(
            self.evasion_regression,
            DefaultEvasionRegressionAttackScorerConfig,
        )
        self.membership_inference = self._coerce_profile(
            self.membership_inference,
            DefaultMembershipInferenceAttackScorerConfig,
        )
        self.attribute_inference = self._coerce_profile(
            self.attribute_inference,
            DefaultAttributeInferenceAttackScorerConfig,
        )
        self.attribute_inference_regression = self._coerce_profile(
            self.attribute_inference_regression,
            DefaultAttributeInferenceRegressionAttackScorerConfig,
        )

    @staticmethod
    def _coerce_profile(profile, default_cls):
        try:
            coerced = coerce_scorer_config(
                profile,
                default_factory=default_cls,
            )
        except ValueError as exc:
            raise TypeError(
                f"Unsupported scorer profile type: {type(profile)}",
            ) from exc
        if coerced is None:
            return default_cls()
        if isinstance(coerced, ScorerDictConfig):
            return coerced
        raise TypeError(f"Unsupported scorer profile type: {type(profile)}")

    @staticmethod
    def _prefix_scores(scores: dict, prefix: str) -> dict:
        prefixed = {}
        for key, value in scores.items():
            prefixed_key = (
                key if str(key).startswith(f"{prefix}_") else f"{prefix}_{key}"
            )
            prefixed[prefixed_key] = value
        return prefixed

    def _score_with_profile(
        self,
        profile: ScorerDictConfig,
        y_true,
        y_pred,
        prefix: str,
        n_samples: int,
        **kwargs,
    ) -> dict:
        raw_scores = profile(
            y_true=y_true,
            y_pred=y_pred,
            mode=None,
            **kwargs,
        )
        prefixed_scores = self._prefix_scores(raw_scores, prefix=prefix)
        return round_scores(prefixed_scores, n_samples=n_samples)

    def _score(
        self,
        attack_kind: Literal["evasion", "membership", "attribute"],
        y_true,
        y_pred,
        attack_size: int,
        ben_pred_labels=None,
        is_classification: Union[bool, None] = None,
        targeted_attribute: Union[str, None] = None,
        attack_generation_time=None,
        sensitive_features=None,
    ):
        if attack_kind == "evasion":
            if is_classification is None:
                is_classification = True
            return self.score_evasion(
                ben_pred_labels=ben_pred_labels,
                adv_pred_labels=y_pred,
                y_true=y_true,
                attack_size=attack_size,
                is_classification=is_classification,
                sensitive_features=sensitive_features,
            )
        if attack_kind == "membership":
            return self.score_membership(
                labels=y_true,
                inferred=y_pred,
                attack_size=attack_size,
                sensitive_features=sensitive_features,
            )
        if attack_kind == "attribute":
            if targeted_attribute is None:
                raise ValueError(
                    "targeted_attribute is required for attribute attack scoring",
                )
            if is_classification is None:
                raise ValueError(
                    "is_classification is required for attribute attack scoring",
                )
            return self.score_attribute(
                target=y_true,
                inferred=y_pred,
                attack_size=attack_size,
                targeted_attribute=targeted_attribute,
                is_classification=is_classification,
                attack_generation_time=attack_generation_time,
                sensitive_features=sensitive_features,
            )
        raise ValueError(f"Unsupported attack scoring kind: {attack_kind}")

    def score_evasion(
        self,
        ben_pred_labels,
        adv_pred_labels,
        y_true,
        attack_size: int,
        is_classification: bool = True,
        sensitive_features=None,
    ):
        start_time = time.process_time()
        profile = self.evasion if is_classification else self.evasion_regression
        score_kwargs = {}
        if is_classification:
            score_kwargs["ben_pred_labels"] = ben_pred_labels
        if sensitive_features is not None:
            score_kwargs["sensitive_features"] = sensitive_features
        score_dict = self._score_with_profile(
            profile=profile,
            y_true=y_true,
            y_pred=adv_pred_labels,
            prefix="evasion",
            n_samples=len(adv_pred_labels),
            **score_kwargs,
        )
        attack_score_time = time.process_time() - start_time
        score_dict["attack_size"] = attack_size
        score_dict["attack_score_time"] = attack_score_time
        return score_dict

    def score_membership(
        self,
        labels,
        inferred,
        attack_size: int,
        sensitive_features=None,
    ):
        start_time = time.process_time()
        score_kwargs = {}
        if sensitive_features is not None:
            score_kwargs["sensitive_features"] = sensitive_features
        score_dict = self._score_with_profile(
            profile=self.membership_inference,
            y_true=labels,
            y_pred=inferred,
            prefix="membership_inference",
            n_samples=len(labels),
            **score_kwargs,
        )
        attack_score_time = time.process_time() - start_time
        score_dict["attack_size"] = attack_size
        score_dict["attack_score_time"] = attack_score_time
        return score_dict

    def score_attribute(
        self,
        target,
        inferred,
        attack_size: int,
        targeted_attribute: str,
        is_classification: bool,
        attack_generation_time=None,
        sensitive_features=None,
    ):
        prefix = f"inferred_{targeted_attribute}"
        start_time = time.process_time()
        score_kwargs = {}
        if sensitive_features is not None:
            score_kwargs["sensitive_features"] = sensitive_features
        if is_classification:
            score_dict = self._score_with_profile(
                profile=self.attribute_inference,
                y_true=target,
                y_pred=inferred,
                prefix=prefix,
                n_samples=len(target),
                **score_kwargs,
            )
        else:
            score_dict = self._score_with_profile(
                profile=self.attribute_inference_regression,
                y_true=target,
                y_pred=inferred,
                prefix=prefix,
                n_samples=len(target),
                **score_kwargs,
            )
        attack_score_time = time.process_time() - start_time
        score_dict["attack_size"] = attack_size
        score_dict["attack_score_time"] = attack_score_time
        if attack_generation_time is not None:
            score_dict["attack_generation_time"] = attack_generation_time
        return score_dict


safe_store(
    group="attack_scorers",
    name="evasion",
    node=DefaultEvasionAttackScorerConfig,
)
safe_store(
    group="attack_scorers",
    name="evasion-regression",
    node=DefaultEvasionRegressionAttackScorerConfig,
)
safe_store(
    group="attack_scorers",
    name="membership-inference",
    node=DefaultMembershipInferenceAttackScorerConfig,
)
safe_store(
    group="attack_scorers",
    name="attribute-inference",
    node=DefaultAttributeInferenceAttackScorerConfig,
)

# Score-chain aliases for attack profile routing in ExperimentConfig.
safe_store(
    group="score",
    name="evasion-classification",
    node={"_deckard_attack_profile": "evasion-classification"},
)
safe_store(
    group="score",
    name="evasion-regression",
    node={"_deckard_attack_profile": "evasion-regression"},
)


@dataclass(eq=False)
class FairlearnEvasionAttackScorerConfig:
    """Per-sensitive-group evasion scorer (classification) via MetricFrame."""

    group_scorers: dict[str, Any] = field(
        default_factory=lambda: {
            "accuracy": ScorerConfig(
                score_name="accuracy",
                score_function="sklearn.metrics.accuracy_score",
            ),
            "f1": ScorerConfig(
                score_name="f1",
                score_function="sklearn.metrics.f1_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
        },
    )


@dataclass(eq=False)
class FairlearnMembershipInferenceAttackScorerConfig:
    """Per-sensitive-group membership inference scorer via MetricFrame."""

    group_scorers: dict[str, Any] = field(
        default_factory=lambda: {
            "accuracy": ScorerConfig(
                score_name="accuracy",
                score_function="sklearn.metrics.accuracy_score",
            ),
            "f1": ScorerConfig(
                score_name="f1",
                score_function="sklearn.metrics.f1_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
        },
    )


@dataclass(eq=False)
class FairlearnAttributeInferenceAttackScorerConfig:
    """Per-sensitive-group attribute inference scorer (classification) via MetricFrame."""

    group_scorers: dict[str, Any] = field(
        default_factory=lambda: {
            "accuracy": ScorerConfig(
                score_name="accuracy",
                score_function="sklearn.metrics.accuracy_score",
            ),
            "f1": ScorerConfig(
                score_name="f1",
                score_function="sklearn.metrics.f1_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
        },
    )


@dataclass(eq=False)
class FairlearnAttributeInferenceRegressionAttackScorerConfig:
    """Per-sensitive-group attribute inference scorer (regression) via MetricFrame."""

    group_scorers: dict[str, Any] = field(
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
        },
    )


@dataclass(eq=False)
class FairlearnAttackScorerConfig(AttackScorerConfig):
    """AttackScorerConfig that computes attack metrics stratified by sensitive group.

    Uses :class:`~deckard.score.fairness.FairlearnScoreDictConfig` profiles for
    each attack type so that metrics (accuracy, f1, mse, …) are computed
    per sensitive group via ``fairlearn.metrics.MetricFrame``.

    Sensitive features must be passed at attack-call time.  In practice,
    :class:`~deckard.attack.base.AttackConfig` injects them automatically
    when the data object exposes ``_sensitive_test`` / ``_sensitive_train``
    (i.e. the data object is a
    :class:`~deckard.data.fairness.FairlearnDataConfig`).
    """

    evasion: Union[ScorerDictConfig, dict, None] = None
    evasion_regression: Union[ScorerDictConfig, dict, None] = None
    membership_inference: Union[ScorerDictConfig, dict, None] = None
    attribute_inference: Union[ScorerDictConfig, dict, None] = None
    attribute_inference_regression: Union[ScorerDictConfig, dict, None] = None

    def __post_init__(self):
        from .fairness import FairlearnScoreDictConfig

        def _fairlearn_profile(field_val, default_group_scorers, base_scorers=None):
            """Return a FairlearnScoreDictConfig, merging any user-supplied overrides."""
            # If user provided a FairlearnScoreDictConfig, use as-is
            if isinstance(field_val, FairlearnScoreDictConfig):
                # Defensive: if scorers or group_scorers are empty, fill with defaults
                if not getattr(field_val, "scorers", None):
                    field_val.scorers = base_scorers or {
                        k: v for k, v in default_group_scorers.items()
                    }
                if not getattr(field_val, "group_scorers", None):
                    field_val.group_scorers = default_group_scorers
                return field_val
            # If user provided a dict, merge with defaults
            if isinstance(field_val, dict):
                scorers = dict(
                    field_val.get(
                        "scorers",
                        base_scorers
                        or {k: v for k, v in default_group_scorers.items()},
                    )
                )
                group_scorers = dict(
                    field_val.get("group_scorers", default_group_scorers)
                )
                return FairlearnScoreDictConfig(
                    scorers=scorers,
                    group_scorers=group_scorers,
                    include_group_by_group=field_val.get(
                        "include_group_by_group", True
                    ),
                    include_group_overall=field_val.get("include_group_overall", True),
                    group_reduction=field_val.get("group_reduction", "difference"),
                )
            # If None, use all defaults
            if field_val is None:
                return FairlearnScoreDictConfig(
                    scorers=base_scorers
                    or {k: v for k, v in default_group_scorers.items()},
                    group_scorers=default_group_scorers,
                    include_group_by_group=True,
                    include_group_overall=True,
                    group_reduction="difference",
                )
            # Fallback: coerce as ScorerDictConfig, then wrap
            coerced = self._coerce_profile(field_val, ScorerDictConfig)
            # Defensive: if scorers or group_scorers are empty, fill with defaults
            if not getattr(coerced, "scorers", None):
                coerced.scorers = base_scorers or {
                    k: v for k, v in default_group_scorers.items()
                }
            if not getattr(coerced, "group_scorers", None):
                coerced.group_scorers = default_group_scorers
            return FairlearnScoreDictConfig(
                scorers=coerced.scorers,
                group_scorers=coerced.group_scorers,
                include_group_by_group=True,
                include_group_overall=True,
                group_reduction="difference",
            )

        # Reasonable defaults for each attack type
        evasion_group = FairlearnEvasionAttackScorerConfig().group_scorers
        membership_group = (
            FairlearnMembershipInferenceAttackScorerConfig().group_scorers
        )
        attribute_group = FairlearnAttributeInferenceAttackScorerConfig().group_scorers
        attribute_reg_group = (
            FairlearnAttributeInferenceRegressionAttackScorerConfig().group_scorers
        )
        evasion_success = {
            "success": ScorerConfig(
                score_name="success",
                score_function="deckard.score.attack.evasion_success_score",
            ),
        }

        self.evasion = _fairlearn_profile(
            self.evasion, evasion_group, base_scorers=evasion_success
        )
        self.evasion_regression = _fairlearn_profile(
            self.evasion_regression, attribute_reg_group
        )
        self.membership_inference = _fairlearn_profile(
            self.membership_inference, membership_group
        )
        self.attribute_inference = _fairlearn_profile(
            self.attribute_inference, attribute_group
        )
        self.attribute_inference_regression = _fairlearn_profile(
            self.attribute_inference_regression, attribute_reg_group
        )


safe_store(
    group="attack_scorers",
    name="fairlearn-attack",
    node=FairlearnAttackScorerConfig,
)
