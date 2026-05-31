"""Configuration for extraction attacks (model stealing)."""

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator
from ..artifacts import ScoreDict
from ..data import DataConfig
from ..frameworks.types import AttackLike, EstimatorLike, MatrixLike
from ..model import ModelConfig
from ..score.base import DefaultClassifierScorerDictConfig

from .base import AttackConfig, AttackFamily, AttackSubFamily
from .poisoning import PoisoningAttackConfig

logger = logging.getLogger(__name__)


@dataclass(eq=False, kw_only=True)
class ExtractionAttackConfig(PoisoningAttackConfig):
    """Configuration for model extraction attacks (model stealing).

    Attributes:
        score_dict: Runtime score payload for extraction metrics.
    """

    @staticmethod
    def _sync_art_classifier_device(classifier: EstimatorLike) -> EstimatorLike:
        """Align ART classifier/preprocessing internals to one device.

        Deep-copying ART classifiers can leave ``_model`` and preprocessing ops on
        different devices. Prefer the wrapped model's device so MPS stays enabled
        when available.
        """
        target_device = getattr(classifier, "_device", None)
        model = getattr(classifier, "_model", None)

        try:
            first_param = next(model.parameters(), None) if model is not None else None
            if first_param is not None and hasattr(first_param, "device"):
                target_device = first_param.device
        except Exception:
            pass

        if target_device is None:
            return classifier

        if hasattr(classifier, "_device"):
            classifier._device = target_device
        if model is not None and hasattr(model, "to"):
            try:
                classifier._model = model.to(target_device)
            except Exception:
                pass

        preprocessing = getattr(classifier, "preprocessing", None)
        if hasattr(preprocessing, "_device"):
            preprocessing._device = target_device
        for op in getattr(classifier, "preprocessing_operations", []) or []:
            if hasattr(op, "_device"):
                op._device = target_device
        return classifier

    def __call__(
        self,
        *,
        data: DataConfig,
        model: ModelConfig | BaseEstimator | EstimatorLike,
        art_model: EstimatorLike,
        attack: AttackLike,
        attack_family: AttackFamily | str,
        attack_sub_family: AttackSubFamily | str,
    ) -> ScoreDict:
        """Run extraction attack runtime handler.

        Args:
            data: Data runtime with sampled train/test/val splits.
            model: User model object/config passed into attack orchestration.
            art_model: ART-wrapped victim model used for extraction.
            attack: Instantiated extraction attack object.
            attack_family: Parsed runtime family; must be ``extraction``.
            attack_sub_family: Parsed sub-family token from attack path.

        Returns:
            Score payload for extraction runtime execution.

        Raises:
            ValueError: If attack type is not extraction.
        """
        _ = attack_sub_family
        if (attack_family or "").lower() != "extraction":
            raise ValueError(
                f"_ExtractionAttackConfig received unsupported attack family: {attack_family}",
            )
        return self.extract(data=data, art_model=art_model, attack=attack)

    @staticmethod
    def _select_extraction_scorer(
        benign_pred: MatrixLike,
        extracted_pred: MatrixLike,
    ) -> tuple[DefaultClassifierScorerDictConfig, bool]:
        """Use full classifier metrics when probabilities are available, else label-only metrics."""
        preds = [np.asarray(benign_pred), np.asarray(extracted_pred)]
        has_probabilities = all(
            AttackConfig._looks_like_probabilities(pred) for pred in preds
        )
        if has_probabilities:
            return DefaultClassifierScorerDictConfig(), True
        label_only = DefaultClassifierScorerDictConfig()
        label_only.scorers.pop("roc_auc", None)
        label_only.scorers.pop("log_loss", None)
        return label_only, False

    def extract(
        self,
        data: DataConfig,
        art_model: EstimatorLike,
        attack: AttackLike,
    ) -> ScoreDict:
        """Execute a model extraction attack and score victim vs extracted classifiers.

        Args:
            data: Runtime dataset and split container.
            art_model: ART-wrapped victim model used for extraction.
            attack: Instantiated extraction attack object.

        Returns:
            Score payload comparing victim and extracted model behavior.

        Raises:
            ValueError: If task/model requirements for extraction are not met.
        """
        task_is_classification = self._infer_task_is_classification(
            data,
            model=art_model,
        )
        if task_is_classification is False:
            raise ValueError(
                "Extraction attacks are only supported for classification tasks.",
            )
        if not self._is_nn_art_classifier(art_model):
            raise ValueError(
                "Extraction attacks currently require a neural-network ART classifier (e.g., PyTorchClassifier).",
            )

        n, x_query, _ = self.get_attack_subset(data, test=False)
        x_query = self._prepare_features_for_art(x_query)

        mode_used, x_eval, y_eval = self._resolve_eval_split(data)
        x_eval = self._prepare_features_for_art(x_eval)
        y_eval = self._normalize_ground_truth(y_eval, is_regression=False)

        thieved_classifier = copy.deepcopy(art_model)
        thieved_classifier = self._sync_art_classifier_device(thieved_classifier)
        thieved_model = getattr(thieved_classifier, "_model", None)
        if thieved_model is not None and hasattr(thieved_model, "apply"):

            def _reset_module_weights(module):
                reset_fn = getattr(module, "reset_parameters", None)
                if callable(reset_fn):
                    reset_fn()

            thieved_model.apply(_reset_module_weights)

        start_time = time.perf_counter()
        extracted_classifier = attack.extract(
            x=x_query,
            thieved_classifier=thieved_classifier,
        )
        self.attack_time = time.perf_counter() - start_time
        logger.info(
            f"Extraction attack training took {self.attack_time} seconds for {n} query samples",
        )

        start_time = time.perf_counter()
        benign_pred = art_model.predict(x_eval)
        extracted_pred = extracted_classifier.predict(x_eval)
        self.attack_prediction_time = time.perf_counter() - start_time
        logger.info(
            f"Extraction prediction took {self.attack_prediction_time} seconds on {mode_used} split",
        )

        benign_labels = self._labels_from_classifier_predictions(benign_pred)
        extracted_labels = self._labels_from_classifier_predictions(extracted_pred)

        start_time = time.perf_counter()
        _, use_proba_metrics = self._select_extraction_scorer(
            benign_pred,
            extracted_pred,
        )
        benign_scores = self._score_comparison(
            y_true=y_eval,
            y_pred=benign_labels,
            stage="benign",
            prefix="benign",
            is_classification=True,
            y_proba=self._to_numpy_array(benign_pred) if use_proba_metrics else None,
            mode=mode_used,
        )
        extracted_scores = self._score_comparison(
            y_true=y_eval,
            y_pred=extracted_labels,
            stage="adversarial",
            prefix="extracted",
            is_classification=True,
            y_proba=(
                self._to_numpy_array(extracted_pred) if use_proba_metrics else None
            ),
            mode=mode_used,
        )
        self.attack_score_time = time.perf_counter() - start_time

        return self._finalize_attack_state(
            attack=extracted_classifier,
            attack_predictions=extracted_pred,
            attacked_labels=y_eval,
            score_dict={
                **benign_scores,
                **extracted_scores,
                "attack_size": n,
                "extraction_mode": mode_used,
            },
        )


    # Note:
    #     Expected family is ``extraction``.
