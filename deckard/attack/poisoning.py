"""Configuration for poisoning attacks (backdoor, trojan)."""

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol, cast

import numpy as np
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierNeuralNetwork
from sklearn.base import BaseEstimator

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..frameworks.types import AttackLike, EstimatorLike, MatrixLike, StringifiedClass
from ..model import ModelConfig
from ..frameworks.pytorch.torch_utils import is_torch_model
from .base import (
    AttackConfig,
    AttackFamily,
    AttackSubFamily,
    AttackTypePlugin,
    AttackMixin,
)

logger = logging.getLogger(__name__)

TorchDeviceLike = str | int | None
PoisonArgValue = str | int | float | bool | None | MatrixLike


class _PoisoningArtModel(Protocol):
    """Minimal ART model contract used by poisoning mixin methods."""

    nb_classes: int | None
    _model: "_TorchLikeModel"
    _device: TorchDeviceLike

    def predict(self, x: MatrixLike) -> MatrixLike:
        """Predict model outputs for the provided matrix-like payload.

        Args:
            x: Feature payload for inference.

        Returns:
            Predicted model outputs.
        """
        ...

    def fit(
        self,
        x: MatrixLike,
        y: MatrixLike,
        **kwargs: PoisonArgValue,
    ) -> None:
        """Fit the model on poisoned samples.

        Args:
            x: Feature payload.
            y: Label payload.
            **kwargs: Optional backend-specific fit parameters.
        """
        ...


class _TorchLikeModel(Protocol):
    """Torch-like model contract supporting device transfer."""

    def to(self, device: TorchDeviceLike) -> "_TorchLikeModel":
        """Move model parameters to the requested device.

        Args:
            device: Target device token.

        Returns:
            Model moved to requested device.
        """
        ...


class _PoisoningAttack(Protocol):
    """Minimal poisoning attack contract for ART attack objects."""

    def poison(
        self,
        *args: PoisonArgValue,
        **kwargs: PoisonArgValue,
    ) -> tuple[MatrixLike, MatrixLike]:
        """Generate poisoned features/labels.

        Args:
            *args: Positional poisoning attack arguments.
            **kwargs: Keyword poisoning attack arguments.

        Returns:
            Poisoned features and labels.
        """
        ...


class PoisoningAttackMixin(AttackMixin):
    """Reusable poisoning attack behavior (backdoor, trojan).

    Attributes:
        score_dict: Runtime score payload for poisoning metrics.
    """

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
        """Dispatch poisoning attack execution for runtime attack family validation.

        Args:
            data: Runtime dataset and split container.
            model: User model configuration or estimator.
            art_model: ART-wrapped model used to run poisoning workflow.
            attack: Instantiated poisoning attack implementation.
            attack_family: Parsed attack family.
            attack_sub_family: Parsed poisoning sub-family.

        Returns:
            Score payload for poisoning runtime execution.

        Raises:
            ValueError: If attack type is not poisoning.
        """
        _ = attack_sub_family
        if (attack_family or "").lower() != "poisoning":
            raise ValueError(
                f"_PoisoningAttackMixin received unsupported attack family: {attack_family}",
            )

        return self.poison(data=data, art_model=art_model, attack=attack)

    def poison(
        self,
        data: DataConfig,
        art_model: _PoisoningArtModel,
        attack: _PoisoningAttack,
    ) -> ScoreDict:
        """Execute poisoning workflow and emit poisoned/benign comparison metrics."""

        attack_name: StringifiedClass = type(attack).__name__.lower()
        if "poisoningattacksvm" in attack_name:
            return self._poison_svm(data=data, art_model=art_model, attack=attack)

        class_source = int(self.attack_params["class_source"])
        class_target = int(self.attack_params["class_target"])
        trigger_index = int(self.attack_params.get("trigger_index", 0))
        poison_fit_params = self.attack_params.get("poison_fit_params", {})

        # ART GradientMatching on macOS can fail with spawned DataLoader workers;
        # force CPU and single-worker loaders for deterministic smoke/integration runs.
        if "gradientmatchingattack" in attack_name:
            try:
                import torch

                art_device = getattr(art_model, "_device", None)
                if getattr(art_device, "type", None) == "mps":
                    if hasattr(art_model, "_model") and hasattr(
                        art_model._model,
                        "to",
                    ):
                        art_model._model = art_model._model.to(torch.device("cpu"))
                    if hasattr(art_model, "_device"):
                        art_model._device = torch.device("cpu")
            except ImportError:
                pass

        x_train = self._to_numpy_array(
            self._prepare_features_for_attack(getattr(data, "X_train")),
            dtype=ART_NUMPY_DTYPE,
        )
        y_train_raw = self._to_numpy_array(
            self._prepare_labels_for_attack(getattr(data, "y_train")),
        )

        mode_used, x_eval_raw, y_eval_raw = self._resolve_eval_split(data)
        x_eval = self._to_numpy_array(
            self._prepare_features_for_attack(x_eval_raw),
            dtype=ART_NUMPY_DTYPE,
        )
        y_eval = self._normalize_ground_truth(y_eval_raw, is_regression=False)
        y_eval_class = self._target_to_class_labels(y_eval_raw)

        source_indices = np.where(y_eval_class == class_source)[0]
        if len(source_indices) == 0:
            fallback_source = int(y_eval_class[0])
            logger.warning(
                "No samples for class_source=%s on %s split; using first available class=%s",
                class_source,
                mode_used,
                fallback_source,
            )
            class_source = fallback_source
            source_indices = np.where(y_eval_class == class_source)[0]

        trigger_pos = trigger_index if trigger_index < len(source_indices) else 0
        trigger_idx = int(source_indices[trigger_pos])
        x_trigger = x_eval[trigger_idx : trigger_idx + 1]

        nb_classes = getattr(art_model, "nb_classes", None)
        if nb_classes is None or int(nb_classes) <= 0:
            nb_classes = int(np.max(y_eval_class)) + 1
        y_trigger = self._one_hot_encode([class_target], nb_classes=int(nb_classes))

        if y_train_raw.ndim == 1 or (
            y_train_raw.ndim == 2 and y_train_raw.shape[1] == 1
        ):
            y_train_for_poison = self._one_hot_encode(
                y_train_raw.reshape(-1),
                nb_classes,
            )
        else:
            y_train_for_poison = y_train_raw

        start_time = time.perf_counter()
        patched_torch_utils_data = None
        patched_dataloader = None
        try:
            if "gradientmatchingattack" in attack_name:
                import torch.utils.data as torch_utils_data

                patched_torch_utils_data = torch_utils_data
                patched_dataloader = torch_utils_data.DataLoader

                def _single_worker_loader(*args, **kwargs):
                    kwargs["num_workers"] = 0
                    return patched_dataloader(*args, **kwargs)

                torch_utils_data.DataLoader = _single_worker_loader

            x_poison, y_poison = attack.poison(
                x_trigger,
                y_trigger,
                x_train,
                y_train_for_poison,
            )
        finally:
            if patched_torch_utils_data is not None and patched_dataloader is not None:
                patched_torch_utils_data.DataLoader = patched_dataloader
        self.attack_time = time.perf_counter() - start_time
        logger.info(
            f"Poison generation took {self.attack_time} seconds for {len(x_poison)} training samples",
        )

        start_time = time.perf_counter()
        benign_pred = art_model.predict(x_eval)
        # Only pass batch_size if art_model is a torch/ART model (not sklearn)
        batch_size = getattr(data, "batch_size", None)
        if batch_size is None:
            batch_size = getattr(getattr(data, "model", None), "fit_params", {}).get(
                "batch_size",
                32,
            )
        poison_fit_params = dict(poison_fit_params) if poison_fit_params else {}
        is_torch_art = hasattr(art_model, "_model") and (
            "torch" in str(type(art_model._model)).lower()
        )
        if is_torch_art:
            poison_fit_params["batch_size"] = batch_size
            art_model.fit(x_poison, y_poison, **poison_fit_params)
        else:
            # For sklearn models, do not pass batch_size
            art_model.fit(x_poison, y_poison)
        poisoned_pred = art_model.predict(x_eval)
        self.attack_prediction_time = time.perf_counter() - start_time
        logger.info(
            f"Poisoned model fit + prediction took {self.attack_prediction_time} seconds on {mode_used} split",
        )

        benign_labels = self._prediction_to_labels(benign_pred, is_regression=False)
        poisoned_labels = self._prediction_to_labels(
            poisoned_pred,
            is_regression=False,
        )

        start_time = time.perf_counter()
        benign_scores = self._score_comparison(
            y_true=y_eval,
            y_pred=benign_labels,
            stage="benign",
            prefix="benign",
            is_classification=True,
            y_proba=benign_pred,
            mode=mode_used,
        )
        poisoned_scores = self._score_comparison(
            y_true=y_eval,
            y_pred=poisoned_labels,
            stage="adversarial",
            prefix="poisoned",
            is_classification=True,
            y_proba=poisoned_pred,
            mode=mode_used,
        )

        trigger_pred = art_model.predict(x_trigger)
        trigger_label = int(self._labels_from_classifier_predictions(trigger_pred)[0])
        self.attack_score_time = time.perf_counter() - start_time

        self.attack_predictions = poisoned_pred
        self.attacked_labels = y_eval
        self.attack = art_model
        merged_scores = self._with_targeted_attack_labels(
            ScoreDict.from_payload(
                {
                    **benign_scores,
                    **poisoned_scores,
                    "poison_attack_target_class": class_target,
                    "poison_attack_source_class": class_source,
                    "poison_trigger_index": trigger_idx,
                    "poison_trigger_predicted_class": trigger_label,
                    "poison_trigger_success": int(trigger_label == class_target),
                    "attack_size": len(x_poison),
                    "poison_mode": mode_used,
                },
            ),
            "poisoning",
        )
        self.score_dict = ScoreDict.from_payload({**self.score_dict, **merged_scores})
        return ScoreDict.from_payload(self.score_dict)

    def _poison_svm(
        self,
        data: DataConfig,
        art_model: _PoisoningArtModel,
        attack: _PoisoningAttack,
    ) -> ScoreDict:
        """Execute an ART PoisoningAttackSVM attack and score benign vs poisoned model accuracy."""
        poison_fit_params = self.attack_params.get("poison_fit_params", {})

        x_train = self._to_numpy_array(
            self._prepare_features_for_attack(getattr(data, "X_train")),
            dtype=ART_NUMPY_DTYPE,
        )
        y_train_class = self._target_to_class_labels(getattr(data, "y_train"))

        mode_used, x_eval_raw, y_eval_raw = self._resolve_eval_split(data)
        x_eval = self._to_numpy_array(
            self._prepare_features_for_attack(x_eval_raw),
            dtype=ART_NUMPY_DTYPE,
        )
        y_eval = self._normalize_ground_truth(y_eval_raw, is_regression=False)
        y_eval_class = self._target_to_class_labels(y_eval_raw)

        nb_classes = int(max(np.max(y_train_class), np.max(y_eval_class))) + 1
        y_train_for_poison = self._one_hot_encode(y_train_class, nb_classes)

        n = min(int(self.attack_size), len(x_eval))
        x_seed = x_eval[:n]
        target_labels = (y_eval_class[:n] + 1) % nb_classes
        y_seed = self._one_hot_encode(target_labels, nb_classes)

        start_time = time.perf_counter()
        x_adv, y_adv = attack.poison(x_seed, y_seed)
        x_adv_arr = np.asarray(x_adv)
        y_adv_arr = np.asarray(y_adv)
        self.attack_time = time.perf_counter() - start_time
        logger.info(
            f"SVM poison generation took {self.attack_time} seconds for {len(x_adv_arr)} generated points",
        )

        x_poison = np.vstack([x_train, x_adv_arr])
        y_poison = np.vstack([y_train_for_poison, y_adv_arr])

        start_time = time.perf_counter()
        benign_pred = art_model.predict(x_eval)
        art_model.fit(x_poison, y_poison, **poison_fit_params)
        poisoned_pred = art_model.predict(x_eval)
        self.attack_prediction_time = time.perf_counter() - start_time
        logger.info(
            f"SVM poisoned model fit + prediction took {self.attack_prediction_time} seconds on {mode_used} split",
        )

        benign_labels = self._prediction_to_labels(benign_pred, is_regression=False)
        poisoned_labels = self._prediction_to_labels(
            poisoned_pred,
            is_regression=False,
        )

        start_time = time.perf_counter()
        benign_scores = self._score_comparison(
            y_true=y_eval,
            y_pred=benign_labels,
            stage="benign",
            prefix="benign",
            is_classification=True,
            y_proba=benign_pred,
            mode=mode_used,
        )
        poisoned_scores = self._score_comparison(
            y_true=y_eval,
            y_pred=poisoned_labels,
            stage="adversarial",
            prefix="poisoned",
            is_classification=True,
            y_proba=poisoned_pred,
            mode=mode_used,
        )
        self.attack_score_time = time.perf_counter() - start_time

        self.attack_predictions = poisoned_pred
        self.attacked_labels = y_eval
        self.attack = art_model
        merged_scores = self._with_targeted_attack_labels(
            ScoreDict.from_payload(
                {
                    **benign_scores,
                    **poisoned_scores,
                    "poisoning_attack_points": int(len(x_adv_arr)),
                    "poison_mode": mode_used,
                    "attack_size": int(len(x_adv_arr)),
                },
            ),
            "poisoning",
        )
        self.score_dict = ScoreDict.from_payload({**self.score_dict, **merged_scores})
        return ScoreDict.from_payload(self.score_dict)

    @staticmethod
    def _is_nn_art_classifier(art_model) -> bool:
        """Return True when the ART estimator appears to be a neural classifier."""
        if isinstance(art_model, ClassifierNeuralNetwork):
            return True
        class_name = type(art_model).__name__.lower()
        if any(
            token in class_name
            for token in (
                "pytorchclassifier",
                "kerasclassifier",
                "tensorflowv2classifier",
            )
        ):
            return True
        model_obj = getattr(art_model, "_model", None)
        if model_obj is None:
            model_obj = getattr(art_model, "model", None)
        return bool(model_obj is not None and is_torch_model(model_obj))

    @classmethod
    def _labels_from_classifier_predictions(cls, predictions):
        _ = cls
        preds = np.asarray(predictions)
        if preds.ndim == 1:
            # Binary classifiers can expose scores/probabilities as a single vector.
            if preds.dtype.kind == "f":
                return (preds >= 0.5).astype(int)
            return preds.astype(int)
        if preds.ndim == 2:
            if preds.shape[1] == 1:
                return (preds.reshape(-1) >= 0.5).astype(int)
            return np.argmax(preds, axis=1)
        raise ValueError(
            f"Unsupported prediction shape for classifier output: {preds.shape}",
        )

    def _resolve_eval_split(self, data: DataConfig):
        requested_mode = self.resolve_mode_for_attack_kind(
            "poisoning",
            attack_sub_family=self.attack_sub_family,
        )

        if requested_mode == "val":
            X_val = getattr(data, "X_val", None)
            y_val = getattr(data, "y_val", None)
            if X_val is not None and y_val is not None:
                return "val", X_val, y_val
            logger.warning(
                "Attack mode='val' requested but validation split is unavailable; falling back to test split.",
            )
        elif requested_mode == "train":
            X_train = getattr(data, "X_train", None)
            y_train = getattr(data, "y_train", None)
            if X_train is not None and y_train is not None:
                return "train", X_train, y_train
            logger.warning(
                "Attack mode='train' requested but training split is unavailable; falling back to test split.",
            )
        X_test = getattr(data, "X_test", None)
        y_test = getattr(data, "y_test", None)
        if X_test is None or y_test is None:
            raise ValueError(
                "Extraction attacks require test features/labels (or val when mode='val').",
            )
        return "test", X_test, y_test


@dataclass(eq=False, kw_only=True)
class PoisoningAttackConfig(PoisoningAttackMixin, AttackConfig):
    """Configuration for poisoning attacks that corrupt training data.

    Note:
        Expected family is ``poisoning``. Runtime behavior is delegated to
        ``PoisoningAttackMixin`` through the default ``AttackTypePlugin``.

    Attributes:
        plugins: Default plugin wiring for ``attack_family='poisoning'``.
    """

    plugins: list = field(
        default_factory=lambda: [
            AttackTypePlugin(
                mixin_type=PoisoningAttackMixin,
                attack_family="poisoning",
            ),
        ],
    )
