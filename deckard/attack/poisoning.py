"""Configuration for poisoning attacks (backdoor, trojan)."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, cast

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


@dataclass(eq=False, kw_only=True)
class PoisoningAttackConfig(AttackConfig):
    """Configuration for poisoning attacks that corrupt training data.

    Attributes:
        score_dict: Runtime score payload for poisoning metrics.
    """

    def __call__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        *,
        data: DataConfig,
        model: ModelConfig | BaseEstimator | EstimatorLike,
        art_model: EstimatorLike,
        attack: AttackLike,
        attack_family: AttackFamily | str,
        attack_sub_family: AttackSubFamily | str,
    ) -> ScoreDict:  # type: ignore[override]
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
                f"_PoisoningAttackConfig received unsupported attack family: {attack_family}",
            )

        return self.poison(data=data, art_model=art_model, attack=attack)

    def _resolve_poison_context(self, data: DataConfig):
        class_source, class_target, trigger_index, poison_fit_params = (
            self._resolve_poison_params()
        )
        mode_used, x_eval_raw, y_eval_raw = self._resolve_eval_split(data)
        x_train = self._to_numpy_array(
            self._prepare_features_for_attack(getattr(data, "X_train")),
            dtype=ART_NUMPY_DTYPE,
        )
        y_train_raw = self._to_numpy_array(
            self._prepare_labels_for_attack(getattr(data, "y_train")),
        )
        x_eval = self._to_numpy_array(
            self._prepare_features_for_attack(x_eval_raw),
            dtype=ART_NUMPY_DTYPE,
        )
        y_eval = self._normalize_ground_truth(y_eval_raw, is_regression=False)
        y_eval_class = self._target_to_class_labels(y_eval_raw)
        return (
            class_source,
            class_target,
            trigger_index,
            poison_fit_params,
            mode_used,
            x_train,
            y_train_raw,
            x_eval,
            y_eval,
            y_eval_class,
        )

    def _select_poison_trigger(
        self,
        *,
        class_source: int,
        class_target: int,
        trigger_index: int,
        mode_used: str,
        x_eval: Any,
        y_eval_class: Any,
    ) -> tuple[int, Any, Any, int, int]:
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
        nb_classes = int(np.max(y_eval_class)) + 1
        y_trigger = self._one_hot_encode([class_target], nb_classes=int(nb_classes))
        return (
            class_source,
            x_eval[trigger_idx : trigger_idx + 1],
            y_trigger,
            trigger_idx,
            nb_classes,
        )

    def _score_poison_predictions(
        self,
        *,
        mode_used: str,
        y_eval: Any,
        benign_pred: Any,
        poisoned_pred: Any,
        extra_scores: dict[str, Any],
    ) -> ScoreDict:
        benign_labels = self._prediction_to_labels(benign_pred, is_regression=False)
        poisoned_labels = self._prediction_to_labels(
            poisoned_pred,
            is_regression=False,
        )
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
        return self._dispatch_attack_scores(
            benign_scores=benign_scores,
            attack_scores=poisoned_scores,
            attack_kind="poisoning",
            extra_scores=extra_scores,
        )

    def _resolve_poison_params(self) -> tuple[int, int, int, dict[str, Any]]:
        attack_params = self.attack_params
        class_source = int(attack_params["class_source"])
        class_target = int(attack_params["class_target"])
        trigger_index = int(attack_params.get("trigger_index", 0))
        poison_fit_params = attack_params.get("poison_fit_params", {})
        return class_source, class_target, trigger_index, poison_fit_params

    @staticmethod
    def _prepare_gradient_matching_attack(
        attack_name: str,
        art_model: _PoisoningArtModel,
    ) -> None:
        if "gradientmatchingattack" not in attack_name:
            return
        try:
            runtime_art_model = cast(Any, art_model)
            art_device = getattr(runtime_art_model, "_device", None)
            if getattr(art_device, "type", None) == "mps":
                if hasattr(runtime_art_model, "_model") and hasattr(
                    runtime_art_model._model,
                    "to",
                ):
                    object.__setattr__(
                        runtime_art_model,
                        "_model",
                        runtime_art_model._model.to("cpu"),
                    )
                if hasattr(runtime_art_model, "_device"):
                    object.__setattr__(runtime_art_model, "_device", "cpu")
        except ImportError:
            pass

    def _build_poison_training_labels(
        self,
        y_train_raw: MatrixLike,
        nb_classes: int,
    ) -> MatrixLike:
        y_train_array = np.asarray(y_train_raw)
        if y_train_array.ndim == 1 or (
            y_train_array.ndim == 2 and y_train_array.shape[1] == 1
        ):
            return self._one_hot_encode(y_train_array.reshape(-1), nb_classes)
        return y_train_array

    def _run_poison_attack(
        self,
        *,
        attack_name: str,
        attack: Any,
        x_trigger: Any,
        y_trigger: Any,
        x_train: Any,
        y_train_for_poison: Any,
    ) -> tuple[Any, Any]:
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

            return attack.poison(
                x_trigger,
                y_trigger,
                x_train,
                y_train_for_poison,
            )
        finally:
            if patched_torch_utils_data is not None and patched_dataloader is not None:
                patched_torch_utils_data.DataLoader = patched_dataloader

    def _finalize_poison_runtime(
        self,
        *,
        poisoned_pred: MatrixLike,
        y_eval: MatrixLike,
        merged_scores: Mapping[str, Any],
        attack_obj: Any,
    ) -> ScoreDict:
        return self._finalize_attack_state(
            attack=attack_obj,
            attack_predictions=poisoned_pred,
            attacked_labels=y_eval,
            score_dict=merged_scores,
        )

    def poison(
        self,
        data: DataConfig,
        art_model: Any,
        attack: Any,
    ) -> ScoreDict:
        """Execute poisoning workflow and emit poisoned/benign comparison metrics.

        Args:
            data: Runtime data config for poisoning workflow inputs.
            art_model: ART estimator wrapper used by the poisoning attack.
            attack: Initialized poisoning attack object.

        Returns:
            Score dictionary containing poisoning runtime outputs.
        """

        attack_name: StringifiedClass = type(attack).__name__.lower()
        if "poisoningattacksvm" in attack_name:
            return self._poison_svm(data=data, art_model=art_model, attack=attack)

        (
            class_source,
            class_target,
            trigger_index,
            poison_fit_params,
            mode_used,
            x_train,
            y_train_raw,
            x_eval,
            y_eval,
            y_eval_class,
        ) = self._resolve_poison_context(data)

        # ART GradientMatching on macOS can fail with spawned DataLoader workers;
        # force CPU and single-worker loaders for deterministic smoke/integration runs.
        self._prepare_gradient_matching_attack(attack_name, art_model)

        (
            class_source,
            x_trigger,
            y_trigger,
            trigger_idx,
            nb_classes,
        ) = self._select_poison_trigger(
            class_source=class_source,
            class_target=class_target,
            trigger_index=trigger_index,
            mode_used=mode_used,
            x_eval=x_eval,
            y_eval_class=y_eval_class,
        )

        y_train_for_poison = self._build_poison_training_labels(
            y_train_raw,
            nb_classes,
        )

        start_time = time.perf_counter()
        x_poison, y_poison = self._run_poison_attack(
            attack_name=attack_name,
            attack=attack,
            x_trigger=x_trigger,
            y_trigger=y_trigger,
            x_train=x_train,
            y_train_for_poison=y_train_for_poison,
        )
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

        start_time = time.perf_counter()
        trigger_pred = art_model.predict(x_trigger)
        trigger_label = int(self._labels_from_classifier_predictions(trigger_pred)[0])
        merged_scores = self._score_poison_predictions(
            mode_used=mode_used,
            y_eval=y_eval,
            benign_pred=benign_pred,
            poisoned_pred=poisoned_pred,
            extra_scores={
                "poison_attack_target_class": class_target,
                "poison_attack_source_class": class_source,
                "poison_trigger_index": trigger_idx,
                "poison_trigger_predicted_class": trigger_label,
                "poison_trigger_success": int(trigger_label == class_target),
                "attack_size": len(x_poison),
                "poison_mode": mode_used,
            },
        )
        self.attack_score_time = time.perf_counter() - start_time
        return self._finalize_poison_runtime(
            poisoned_pred=poisoned_pred,
            y_eval=y_eval,
            merged_scores=merged_scores,
            attack_obj=art_model,
        )

    def _poison_svm(
        self,
        data: DataConfig,
        art_model: Any,
        attack: Any,
    ) -> ScoreDict:
        """Execute an ART PoisoningAttackSVM attack and score benign vs poisoned model accuracy."""
        poison_fit_params = dict(self.attack_params.get("poison_fit_params", {}) or {})

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

        start_time = time.perf_counter()
        merged_scores = self._score_poison_predictions(
            mode_used=mode_used,
            y_eval=y_eval,
            benign_pred=benign_pred,
            poisoned_pred=poisoned_pred,
            extra_scores={
                "poisoning_attack_points": int(len(x_adv_arr)),
                "poison_mode": mode_used,
                "attack_size": int(len(x_adv_arr)),
            },
        )
        self.attack_score_time = time.perf_counter() - start_time
        return self._finalize_poison_runtime(
            poisoned_pred=poisoned_pred,
            y_eval=y_eval,
            merged_scores=merged_scores,
            attack_obj=art_model,
        )

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

    # Note:
    #     Expected family is ``poisoning``.
