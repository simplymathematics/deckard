"""Attack-specific scoring helpers and default scorer configuration."""

from dataclasses import dataclass, field
import time
from typing import Dict, Literal, Union

from sklearn.metrics import accuracy_score

from ..utils import ConfigBase, round_scores
from .base import ScorerConfig, ScorerDictConfig, safe_store

__all__ = [
    "evasion_success_score",
    "DefaultEvasionAttackScorerConfig",
    "DefaultEvasionRegressionAttackScorerConfig",
    "DefaultMembershipInferenceAttackScorerConfig",
    "DefaultAttributeInferenceAttackScorerConfig",
    "DefaultAttributeInferenceRegressionAttackScorerConfig",
    "AttackScorerConfig",
]


def evasion_success_score(y_true, y_pred, ben_pred_labels=None, **kwargs):
    """Compute evasion success as one minus benign/adversarial agreement."""
    if ben_pred_labels is None:
        raise ValueError("ben_pred_labels are required for evasion_success scoring")
    return float(1 - accuracy_score(ben_pred_labels, y_pred))


@dataclass(eq=False)
class DefaultEvasionAttackScorerConfig(ScorerDictConfig):
    """Default scorer set for evasion attack evaluation."""

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
            "f1-score": ScorerConfig(
                score_name="f1-score",
                score_function="sklearn.metrics.f1_score",
                score_params={"average": "weighted", "zero_division": 0},
            ),
            "success": ScorerConfig(
                score_name="success",
                score_function="deckard.score.attack.evasion_success_score",
            ),
        },
    )


@dataclass(eq=False)
class DefaultEvasionRegressionAttackScorerConfig(ScorerDictConfig):
    """Default scorer set for evasion attacks against regression models."""

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
class DefaultMembershipInferenceAttackScorerConfig(ScorerDictConfig):
    """Default scorer set for membership inference attack evaluation."""

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
class DefaultAttributeInferenceAttackScorerConfig(ScorerDictConfig):
    """Default scorer set for categorical attribute inference evaluation."""

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
class DefaultAttributeInferenceRegressionAttackScorerConfig(ScorerDictConfig):
    """Default scorer set for continuous attribute inference evaluation."""

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
        if profile is None:
            return default_cls()
        if isinstance(profile, ScorerDictConfig):
            return profile
        if isinstance(profile, dict):
            if "scorers" in profile:
                return ScorerDictConfig(**profile)
            return ScorerDictConfig(scorers=profile)
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
            )
        if attack_kind == "membership":
            return self.score_membership(
                labels=y_true,
                inferred=y_pred,
                attack_size=attack_size,
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
            )
        raise ValueError(f"Unsupported attack scoring kind: {attack_kind}")

    def score_evasion(
        self,
        ben_pred_labels,
        adv_pred_labels,
        y_true,
        attack_size: int,
        is_classification: bool = True,
    ):
        start_time = time.process_time()
        profile = self.evasion if is_classification else self.evasion_regression
        score_kwargs = {}
        if is_classification:
            score_kwargs["ben_pred_labels"] = ben_pred_labels
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

    def score_membership(self, labels, inferred, attack_size: int):
        start_time = time.process_time()
        score_dict = self._score_with_profile(
            profile=self.membership_inference,
            y_true=labels,
            y_pred=inferred,
            prefix="membership_inference",
            n_samples=len(labels),
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
    ):
        prefix = f"inferred_{targeted_attribute}"
        start_time = time.process_time()
        if is_classification:
            score_dict = self._score_with_profile(
                profile=self.attribute_inference,
                y_true=target,
                y_pred=inferred,
                prefix=prefix,
                n_samples=len(target),
            )
        else:
            score_dict = self._score_with_profile(
                profile=self.attribute_inference_regression,
                y_true=target,
                y_pred=inferred,
                prefix=prefix,
                n_samples=len(target),
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
