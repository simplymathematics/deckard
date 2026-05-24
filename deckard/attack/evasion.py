"""Configuration for evasion attacks (adversarial examples)."""

import logging
import time
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from art.config import ART_NUMPY_DTYPE
from sklearn.base import BaseEstimator

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..frameworks.types import ArrayLike, AttackLike, EstimatorLike, StringifiedClass
from ..model import ModelConfig
from ..frameworks.pytorch.torch_utils import (
    is_tensor,
    tensor_to_numpy,
)
from .base import AttackConfig, AttackTypePlugin, AttackMixin, _sensitive_slice

logger = logging.getLogger(__name__)


class EvasionAttackMixin(AttackMixin):
    """Reusable evasion attack behavior."""

    # Declared for static analyzers
    attack_size: int
    attack_time: Union[float, None]
    attack_prediction_time: Union[float, None]
    attack_predictions: ArrayLike | None
    attacked_labels: ArrayLike | None
    score_dict: ScoreDict

    def __call__(
        self,
        *,
        data: DataConfig,
        model: ModelConfig | BaseEstimator | EstimatorLike,
        art_model: EstimatorLike,
        attack: AttackLike,
        attack_type: StringifiedClass,
        attack_subtype: StringifiedClass,
    ) -> ScoreDict:
        """Dispatch evasion attack execution for runtime attack family validation.

        Args:
            data: Runtime dataset and split container.
            model: User model configuration or estimator.
            art_model: ART-wrapped model used for prediction and attack evaluation.
            attack: Instantiated evasion attack implementation.
            attack_type: Parsed attack family.
            attack_subtype: Parsed evasion subtype.
        """
        if (attack_type or "").lower() != "evasion":
            raise ValueError(
                f"_EvasionAttackMixin received unsupported attack type: {attack_type}",
            )
        return self.evade(data, art_model, attack)

    def evade(
        self,
        data: DataConfig,
        art_model: EstimatorLike,
        attack: AttackLike,
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
        """

        start_time = time.perf_counter()
        active_mode = self.resolve_mode_for_attack_kind("evasion")
        n, x_subset, y_subset = self.get_attack_subset(
            data,
            test=(active_mode != "train"),
        )
        x_subset = self._prepare_features_for_attack(x_subset)
        y_subset = self._prepare_labels_for_attack(y_subset)
        x_subset_art = self._prepare_features_for_art(x_subset)
        if not isinstance(y_subset, (list, np.ndarray)) and not is_tensor(y_subset):
            raise TypeError(
                f"Expected labels to be a list, numpy array, or tensor. Got {type(y_subset)}",
            )
        ben_preds = art_model.predict(x_subset_art)
        is_regression = self._is_regression_prediction_output(
            y_subset,
            ben_preds,
        )
        ben_pred_labels = self._prediction_to_labels(
            ben_preds,
            is_regression=is_regression,
        )
        if is_tensor(ben_pred_labels):
            ben_pred_labels = tensor_to_numpy(
                ben_pred_labels,
                dtype=ART_NUMPY_DTYPE,
            )
        if "AdversarialPatch" in str(type(attack)):
            # Special handling for AdversarialPatch attack
            patches = attack.generate(x=x_subset_art, y=ben_pred_labels)
            # Caclulate the scale of the patch, relative to the input size
            input_shape = x_subset_art[0].shape[
                1:
            ]  # Exclude batch dimension, channel dimension
            patch_shape = patches[0].shape[
                1:
            ]  # Exclude batch dimension, channel dimension
            # Assume that the patch is square (required by the attack)
            # Calculate the scale based on the larger input_dimension
            scale = max(
                patch_shape[0] / input_shape[0],
                patch_shape[1] / input_shape[1],
            )
            X_test_adv = attack.apply_patch(x_subset_art, scale=scale)
        else:
            X_test_adv = attack.generate(x=x_subset_art)
        end_time = time.perf_counter()
        self.attack_time = end_time - start_time
        logger.info(
            f"Evasion attack took {self.attack_time} seconds for {n} samples",
        )
        start_time = time.perf_counter()
        adv_pred = art_model.predict(X_test_adv)
        self.attack_predictions = adv_pred
        self.attacked_labels = y_subset
        # adv_pred_labels = adv_pred.argmax(axis=1)
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

        score_dict = self._score(
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
        logger.info(
            f"Attack scoring took {self.attack_score_time} seconds for {len(adv_pred_labels)} samples and {len(self.score_dict)} scores.",
        )
        self.score_dict = ScoreDict.from_payload(
            {**self.score_dict, **benign_scores, **score_dict},
        )
        for score in self.score_dict:
            logger.info(f"{score}: {self.score_dict[score]}")
        self.attack = adv_pred
        return ScoreDict.from_payload(self.score_dict)


@dataclass(eq=False, kw_only=True)
class EvasionAttackConfig(EvasionAttackMixin, AttackConfig):
    """Configuration for evasion attacks that generate adversarial examples.

    Initialization params
    ---------------------
    attack_type : str
        Attack family path inherited from ``AttackConfig``. Expected family is
        ``evasion``.
    attack_params : dict[str, Any]
        Constructor kwargs forwarded to resolved ART evasion attack classes.
    plugins : list[AttackTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``AttackTypePlugin`` configured with:
        ``mixin_type: type = _EvasionAttackMixin`` and
        ``attack_type: str = 'evasion'``.

    Runtime params
    --------------
    _EvasionAttackMixin.__call__(self, *, data: Any, model: Any, art_model: Any, attack: Any, attack_type: str, attack_subtype: str) -> ScoreDict
        Runtime dispatch entrypoint invoked by ``AttackConfig.__call__``.
    _EvasionAttackMixin.evade(self, data: Any, art_model: Any, attack: Any) -> ScoreDict
        Generates adversarial examples and returns score payload.
    """

    plugins: list = field(
        default_factory=lambda: [
            AttackTypePlugin(
                mixin_type=EvasionAttackMixin,
                attack_type="evasion",
            ),
        ],
    )
