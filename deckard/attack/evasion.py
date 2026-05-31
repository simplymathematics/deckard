"""Configuration for evasion attacks (adversarial examples)."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from art.config import ART_NUMPY_DTYPE
from sklearn.base import BaseEstimator

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..frameworks.types import EstimatorLike
from ..model import ModelConfig
from ..frameworks.pytorch.torch_utils import (
    is_tensor,
    tensor_to_numpy,
)
from .base import (
    AttackConfig,
    AttackFamily,
    AttackSubFamily,
    _sensitive_slice,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False, kw_only=True)
class EvasionAttackConfig(AttackConfig):
    """Configuration for evasion attacks that generate adversarial examples.

    Attributes:
        attack_size: Number of samples used for evasion attack evaluation.
        attack_time: Runtime duration for adversarial example generation.
        attack_prediction_time: Runtime duration for adversarial prediction.
        attack_predictions: Stored adversarial outputs/predictions.
        attacked_labels: Labels associated with attacked samples.
        score_dict: Runtime score payload for evasion metrics.
    """

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        data: DataConfig,
        model: ModelConfig | BaseEstimator | EstimatorLike,
        art_model: Any,
        attack: Any,
        attack_family: AttackFamily | str,
        attack_sub_family: AttackSubFamily | str,
    ) -> ScoreDict:  # type: ignore[override]
        """Dispatch evasion attack execution for runtime attack family validation.

        Args:
            data: Runtime dataset and split container.
            model: User model configuration or estimator.
            art_model: ART-wrapped model used for prediction and attack evaluation.
            attack: Instantiated evasion attack implementation.
            attack_family: Parsed attack family.
            attack_sub_family: Parsed evasion sub-family.

        Returns:
            Score payload for evasion attack execution.

        Raises:
            ValueError: If attack family is not evasion.
        """
        _ = attack_sub_family
        if (attack_family or "").lower() != "evasion":
            raise ValueError(
                f"_EvasionAttackConfig received unsupported attack family: {attack_family}",
            )
        return self.evade(data, art_model, attack)

    def _resolve_evasion_context(self, data: DataConfig):
        active_mode = self.resolve_mode_for_attack_kind("evasion")
        n, x_subset, y_subset = self.get_attack_subset(
            data,
            test=(active_mode != "train"),
        )
        x_subset = self._prepare_features_for_attack(x_subset)
        y_subset = self._prepare_labels_for_attack(y_subset)
        x_subset_art = self._prepare_features_for_art(x_subset)
        return active_mode, n, x_subset_art, y_subset

    def _predict_benign_labels(self, art_model: Any, x_subset_art: Any, y_subset: Any):
        ben_preds = art_model.predict(x_subset_art)
        is_regression = self._is_regression_prediction_output(y_subset, ben_preds)
        ben_pred_labels = self._prediction_to_labels(
            ben_preds,
            is_regression=is_regression,
        )
        if is_tensor(ben_pred_labels):
            ben_pred_labels = tensor_to_numpy(
                ben_pred_labels,
                dtype=ART_NUMPY_DTYPE,
            )
        return ben_preds, ben_pred_labels, is_regression

    def _generate_evasion_examples(
        self,
        attack: Any,
        x_subset_art: Any,
        ben_pred_labels: Any,
    ):
        if "AdversarialPatch" in str(type(attack)):
            patches = attack.generate(x=x_subset_art, y=ben_pred_labels)
            input_shape = x_subset_art[0].shape[1:]
            patch_shape = patches[0].shape[1:]
            scale = max(
                patch_shape[0] / input_shape[0],
                patch_shape[1] / input_shape[1],
            )
            return attack.apply_patch(x_subset_art, scale=scale)
        return attack.generate(x=x_subset_art)

    def _score_evasion_attack(
        self,
        *,
        data: DataConfig,
        n: int,
        active_mode: str,
        y_subset: Any,
        ben_preds: Any,
        ben_pred_labels: Any,
        adv_pred: Any,
        adv_pred_labels: Any,
        is_regression: bool,
    ) -> ScoreDict:
        y_test_numeric = self._normalize_ground_truth(
            y_subset,
            is_regression=is_regression,
        )
        benign_scores = self._score_comparison(
            y_true=y_test_numeric,
            y_pred=ben_pred_labels,
            stage="benign",
            prefix="benign",
            is_classification=not is_regression,
            y_proba=None if is_regression else ben_preds,
            mode=active_mode,
            sensitive_features=_sensitive_slice(
                getattr(data, "_sensitive_test", None),
                n,
            ),
        )
        attack_scores = self._score(
            attack_kind="evasion",
            y_true=y_test_numeric,
            y_pred=adv_pred_labels,
            ben_pred_labels=ben_pred_labels,
            is_classification=not is_regression,
            sensitive_features=_sensitive_slice(
                getattr(data, "_sensitive_test", None),
                n,
            ),
        )
        merged_attack_scores = self._dispatch_attack_scores(
            benign_scores=benign_scores,
            attack_scores=attack_scores,
            attack_kind="evasion",
        )
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        return merged_attack_scores

    def evade(
        self,
        data: DataConfig,
        art_model: Any,
        attack: Any,
    ) -> ScoreDict:
        """
        Executes an evasion attack on a given dataset using the specified ART model and attack method.

        This method assumes a classification task and generates adversarial examples from a subset of the test data.
        It measures and logs the time taken for both the attack generation and adversarial prediction steps.
        The method then evaluates the attack by comparing benign and adversarial predictions against the true labels,
        and stores the attack results and scores.

        Args:
            data: The dataset containing features and labels.
            art_model: ART model used for predictions.
            attack: ART attack object used to generate adversarial examples.

        Returns:
            Score payload containing attack evaluation metrics.

        Raises:
            TypeError: If runtime labels have unsupported type.
            ValueError: If attack runtime payloads are invalid.
        """

        active_mode, n, x_subset_art, y_subset = self._resolve_evasion_context(data)
        if not isinstance(y_subset, (list, np.ndarray)) and not is_tensor(y_subset):
            raise TypeError(
                f"Expected labels to be a list, numpy array, or tensor. Got {type(y_subset)}",
            )
        start_time = time.perf_counter()
        ben_preds, ben_pred_labels, is_regression = self._predict_benign_labels(
            art_model,
            x_subset_art,
            y_subset,
        )
        X_test_adv = self._generate_evasion_examples(
            attack,
            x_subset_art,
            ben_pred_labels,
        )
        end_time = time.perf_counter()
        self.attack_time = end_time - start_time
        logger.info(
            f"Evasion attack took {self.attack_time} seconds for {n} samples",
        )
        start_time = time.perf_counter()
        adv_pred = art_model.predict(X_test_adv)
        end_time = time.perf_counter()
        self.attack_prediction_time = end_time - start_time
        logger.info(
            f"Adversarial prediction took {self.attack_prediction_time} seconds for {n} samples",
        )
        adv_pred_labels = self._prediction_to_labels(
            adv_pred,
            is_regression=is_regression,
        )
        self.score_y_pred = adv_pred_labels
        self.score_y_proba = adv_pred
        merged_attack_scores = self._score_evasion_attack(
            data=data,
            n=n,
            active_mode=active_mode,
            y_subset=y_subset,
            ben_preds=ben_preds,
            ben_pred_labels=ben_pred_labels,
            adv_pred=adv_pred,
            adv_pred_labels=adv_pred_labels,
            is_regression=is_regression,
        )
        for score in merged_attack_scores:
            logger.info(f"{score}: {merged_attack_scores[score]}")
        return self._finalize_attack_state(
            attack=adv_pred,
            attack_predictions=adv_pred,
            attacked_labels=y_subset,
            score_dict=merged_attack_scores,
            score_y_pred=adv_pred_labels,
            score_y_proba=adv_pred,
        )

    # Note:
    #     Expected family is ``evasion``.
