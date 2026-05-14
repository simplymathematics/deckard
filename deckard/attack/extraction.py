"""Configuration for extraction attacks (model stealing)."""

import copy
import time
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from .base import AttackConfig, AttackTypePlugin
from .poisoning import _PoisoningAttackMixin



logger = logging.getLogger(__name__)


class _ExtractionAttackMixin(_PoisoningAttackMixin):
    """Reusable extraction attack behavior (model stealing)."""

    def __call__(
        self,
        *,
        data,
        model,
        art_model,
        attack,
        attack_type: str,
        attack_subtype: str,
    ) -> dict:
        """Run extraction attack runtime handler.

        Parameters
        ----------
        data : Any
            Data runtime with sampled train/test/val splits.
        model : Any
            User model object/config passed into attack orchestration.
        art_model : Any
            ART-wrapped victim model used for extraction.
        attack : Any
            Instantiated extraction attack object.
        attack_type : str
            Parsed runtime family; must be ``extraction`` for this mixin.
        attack_subtype : str
            Parsed subtype token from attack path.
        """
        if (attack_type or "").lower() != "extraction":
            raise ValueError(
                f"_ExtractionAttackMixin received unsupported attack type: {attack_type}",
            )
        return self._extract(data=data, art_model=art_model, attack=attack)

    def _extract(self, data, art_model, attack) -> dict:
        """Execute a model extraction attack and score victim vs extracted classifiers."""
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
        thieved_model = getattr(thieved_classifier, "_model", None)
        if thieved_model is not None and hasattr(thieved_model, "apply"):

            def _reset_module_weights(module):
                reset_fn = getattr(module, "reset_parameters", None)
                if callable(reset_fn):
                    reset_fn()

            thieved_model.apply(_reset_module_weights)

        start_time = time.process_time()
        extracted_classifier = attack.extract(
            x=x_query,
            thieved_classifier=thieved_classifier,
        )
        self.attack_time = time.process_time() - start_time
        logger.info(
            f"Extraction attack training took {self.attack_time} seconds for {n} query samples",
        )

        start_time = time.process_time()
        benign_pred = art_model.predict(x_eval)
        extracted_pred = extracted_classifier.predict(x_eval)
        self.attack_prediction_time = time.process_time() - start_time
        logger.info(
            f"Extraction prediction took {self.attack_prediction_time} seconds on {mode_used} split",
        )

        benign_labels = self._labels_from_classifier_predictions(benign_pred)
        extracted_labels = self._labels_from_classifier_predictions(extracted_pred)

        start_time = time.process_time()
        classification_scorer, use_proba_metrics = self._select_extraction_scorer(
            benign_pred,
            extracted_pred,
        )
        benign_kwargs = {
            "y_true": y_eval,
            "y_pred": benign_labels,
            "mode": None,
        }
        extracted_kwargs = {
            "y_true": y_eval,
            "y_pred": extracted_labels,
            "mode": None,
        }
        if use_proba_metrics:
            benign_kwargs["y_proba"] = self._to_numpy_array(benign_pred)
            extracted_kwargs["y_proba"] = self._to_numpy_array(extracted_pred)

        benign_scores = classification_scorer(
            y_true=y_eval,
            **{k: v for k, v in benign_kwargs.items() if k != "y_true"},
        )
        extracted_scores = classification_scorer(
            y_true=y_eval,
            **{k: v for k, v in extracted_kwargs.items() if k != "y_true"},
        )
        self.attack_score_time = time.process_time() - start_time

        prefixed_benign = {f"benign_{k}": v for k, v in benign_scores.items()}
        prefixed_extracted = {f"extracted_{k}": v for k, v in extracted_scores.items()}
        self.attack_predictions = extracted_pred
        self.attacked_labels = y_eval
        self.attack = extracted_classifier
        self.score_dict = {
            **self.score_dict,
            **prefixed_benign,
            **prefixed_extracted,
            "attack_size": n,
            "extraction_mode": mode_used,
        }
        return self.score_dict


@dataclass(eq=False)
class ExtractionAttackConfig(_ExtractionAttackMixin, AttackConfig):
    """Configuration for model extraction attacks (model stealing).

    Initialization params
    ---------------------
    attack_type : str
        Attack family path inherited from ``AttackConfig``. Expected family is
        ``extraction``.
    attack_params : dict[str, Any]
        Constructor kwargs and runtime controls used by extraction attacks.
    init_params : dict[str, Any]
        Metadata-only declaration payload for class/type/library docs.
    plugins : list[AttackTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``AttackTypePlugin`` configured with:
        ``mixin_type: type = _ExtractionAttackMixin`` and
        ``attack_type: str = 'extraction'``.

    Runtime params
    --------------
    _ExtractionAttackMixin.__call__(self, *, data: Any, model: Any, art_model: Any, attack: Any, attack_type: str, attack_subtype: str) -> dict
        Runtime dispatch entrypoint invoked by ``AttackConfig.__call__``.
    _ExtractionAttackMixin._extract(self, data: Any, art_model: Any, attack: Any) -> dict
        Executes extraction flow and returns score payload.
    """

    plugins: list = field(
        default_factory=lambda: [
            AttackTypePlugin(
                mixin_type=_ExtractionAttackMixin,
                attack_type="extraction",
            )
        ]
    )

