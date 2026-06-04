"""Configuration for inference attacks (membership, attribute, model inversion)."""

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from art.config import ART_NUMPY_DTYPE
from numpy.exceptions import AxisError
from omegaconf import ListConfig, OmegaConf

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..model import ModelConfig
from ..orchestration import resolve_attack_split_payload
from ..types import AttackLike, EstimatorLike
from .base import (
    AttackConfig,
    AttackFamily,
    AttackSubFamily,
    _sensitive_slice,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False, kw_only=True)
class InferenceAttackConfig(AttackConfig):
    """Configuration for privacy inference attacks.

    Attributes:
        targeted_attribute: Attribute name/index used by attribute inference flows.
        score_dict: Runtime score payload for inference metrics.
    """

    targeted_attribute: str

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
        """Dispatch inference attack execution for supported inference subtypes.

        Args:
            data: Runtime dataset and split container.
            model: User model configuration or estimator.
            art_model: ART-wrapped model used for inference attacks.
            attack: Instantiated inference attack implementation.
            attack_family: Parsed attack family.
            attack_sub_family: Parsed attack sub-family.

        Returns:
            Score payload for the selected inference subtype.

        Raises:
            ValueError: If attack family/subtype is unsupported.
        """
        if (attack_family or "").lower() != "inference":
            raise ValueError(
                f"_InferenceAttackConfig received unsupported attack family: {attack_family}",
            )
        return self.infer(
            data=data,
            art_model=art_model,
            attack=attack,
            attack_sub_family=attack_sub_family,
        )

    def infer(
        self,
        *,
        data: DataConfig,
        art_model: EstimatorLike,
        attack: AttackLike,
        attack_sub_family: AttackSubFamily | str,
    ) -> ScoreDict:
        """Public type-mirroring dispatcher for inference attack subtypes.

        Args:
            data: Runtime dataset and split container.
            art_model: ART-wrapped model used for inference attacks.
            attack: Instantiated inference attack implementation.
            attack_sub_family: Parsed inference sub-family token.

        Returns:
            Score payload for the selected inference subtype.

        Raises:
            ValueError: If attack subtype is unsupported.
        """
        subtype = (attack_sub_family or "").lower()
        if subtype == "membership_inference":
            return self.infer_membership(data=data, attack=attack)
        if subtype == "attribute_inference":
            assert (
                self.targeted_attribute is not None
            ), "targeted_attribute must be specified for inference attacks"
            return self.infer_attribute(
                data=data,
                art_model=art_model,
                attack=attack,
                targeted_attribute=self.targeted_attribute,
            )
        if subtype == "model_inversion":
            return self.infer_model_inversion(data=data, attack=attack)
        if subtype == "reconstruction":
            return self.infer_database_reconstruction(data=data, attack=attack)
        raise ValueError(
            f"Unsupported inference attack sub-family: {attack_sub_family}",
        )

    def inference(
        self,
        *,
        data: DataConfig,
        art_model: EstimatorLike,
        attack: AttackLike,
        attack_sub_family: AttackSubFamily | str,
    ) -> ScoreDict:
        """Backward-compatible noun-mode alias for ``infer``.

        Args:
            data: Runtime dataset and split container.
            art_model: ART-wrapped model used for inference attacks.
            attack: Instantiated inference attack implementation.
            attack_sub_family: Parsed inference sub-family token.

        Returns:
            Score payload for the selected inference subtype.
        """
        return self.infer(
            data=data,
            art_model=art_model,
            attack=attack,
            attack_sub_family=attack_sub_family,
        )

    def membership_inference(
        self,
        data: DataConfig,
        attack: AttackLike,
    ) -> ScoreDict:
        """Public subtype-mirroring alias for membership inference execution.

        Args:
            data: Runtime dataset and split container.
            attack: Instantiated membership inference attack implementation.

        Returns:
            Score payload for membership inference execution.
        """
        return self.infer_membership(data=data, attack=attack)

    def attribute_inference(
        self,
        data: DataConfig,
        art_model: EstimatorLike,
        attack: AttackLike,
        targeted_attribute: str | list[str] | ListConfig,
    ) -> ScoreDict:
        """Public subtype-mirroring alias for attribute inference execution.

        Args:
            data: Runtime dataset and split container.
            art_model: ART-wrapped model used for attribute inference.
            attack: Instantiated attribute inference attack implementation.
            targeted_attribute: Target attribute name(s) to reconstruct.

        Returns:
            Score payload for attribute inference execution.
        """
        return self.infer_attribute(
            data=data,
            art_model=art_model,
            attack=attack,
            targeted_attribute=targeted_attribute,
        )

    def model_inversion(self, data: DataConfig, attack: AttackLike) -> ScoreDict:
        """Public subtype-mirroring alias for model inversion execution.

        Args:
            data: Runtime dataset and split container.
            attack: Instantiated model inversion attack implementation.

        Returns:
            Score payload for model inversion execution.
        """
        return self.infer_model_inversion(data=data, attack=attack)

    def reconstruct(self, data: DataConfig, attack: AttackLike) -> ScoreDict:
        """Public subtype-mirroring alias for reconstruction execution.

        Args:
            data: Runtime dataset and split container.
            attack: Instantiated reconstruction attack implementation.

        Returns:
            Score payload for reconstruction execution.
        """
        return self.infer_database_reconstruction(data=data, attack=attack)

    def infer_attribute(
        self,
        data: DataConfig,
        art_model: EstimatorLike,
        attack: AttackLike,
        targeted_attribute: str | list[str] | ListConfig,
    ) -> ScoreDict:
        """Infer held-out attribute values from model outputs.

        Args:
            data: Runtime data config providing attack source features.
            art_model: ART-wrapped estimator used for attribute inference.
            attack: Instantiated attribute-inference attack implementation.
            targeted_attribute: Column name or column-name list to reconstruct.

        Returns:
            Score payload for the reconstructed attribute predictions.

        Raises:
            ValueError: If targeted attributes are invalid or missing.
        """
        assert hasattr(data, "X_train") and hasattr(
            data,
            "y_train",
        ), "DataConfig must have X_train, y_train attributes. Please ensure data() has been called."
        targeted_attribute_string = str(targeted_attribute)
        if isinstance(targeted_attribute, str):
            assert targeted_attribute in data.X_test.columns, (
                f"Targeted attribute '{targeted_attribute}' not found in test data columns.",
            )
        else:
            assert isinstance(
                targeted_attribute,
                (list, ListConfig),
            ), "targeted attribute must be a string or a list of strings"
            if isinstance(targeted_attribute, ListConfig):
                targeted_attribute = OmegaConf.to_container(targeted_attribute)
            if not isinstance(targeted_attribute, list):
                targeted_attribute = [targeted_attribute]
            targeted_attribute_string = ""
            for attr in targeted_attribute:
                try:
                    assert attr in data.X_test.columns
                    if len(targeted_attribute_string) > 0:
                        targeted_attribute_string += f"-{attr}"
                    else:
                        targeted_attribute_string = f"{attr}"
                except AssertionError:
                    raise ValueError(
                        f"Targeted attribute '{attr}' not found in test data columns.",
                    )
        active_mode = self.resolve_mode_for_attack_kind(
            "attribute",
            attack_sub_family=self.attack_sub_family,
        )
        if active_mode == "val":
            X_source = getattr(data, "X_val", None)
            if X_source is None:
                X_source = data.X_test
        elif active_mode == "train":
            X_source = data.X_train
        else:
            X_source = data.X_test

        X_test = X_source.copy()
        target = X_test[targeted_attribute].copy()
        X_test_subset = X_test.iloc[: self.attack_size, :].copy().values
        target = target[: self.attack_size].values

        X_test_subset_without_feature = (
            X_test.drop(
                columns=targeted_attribute,
            )
            .copy()
            .iloc[: self.attack_size, :]
            .values
        )
        assert (
            len(X_test_subset) == self.attack_size
        ), f"Test set size {len(X_test_subset)} does not match attack_size {self.attack_size}"
        start_time = time.perf_counter()
        attack.fit(x=X_test_subset)
        attack_time = time.perf_counter() - start_time
        self.attack_time = attack_time
        preds = self._prediction_to_labels(
            art_model.predict(X_test_subset),
            is_regression=False,
        ).reshape(-1, 1)
        possible_values = np.unique(target, axis=0)
        if isinstance(possible_values, np.ndarray):
            possible_values = possible_values.tolist()
        if (
            isinstance(possible_values, list)
            and len(possible_values) > 0
            and isinstance(possible_values[0], list)
            and len(possible_values[0]) == 1
        ):
            possible_values = [v[0] for v in possible_values]
        start_time = time.perf_counter()
        preds = np.array(preds, dtype=ART_NUMPY_DTYPE)
        X_test_subset_without_feature = np.array(
            X_test_subset_without_feature,
            dtype=ART_NUMPY_DTYPE,
        )
        inferred = attack.infer(
            x=X_test_subset_without_feature,
            pred=preds,
            values=possible_values,
        )
        end_time = time.perf_counter()
        is_classification = not attack._is_continuous
        inferred = self._normalize_inferred_output(inferred)
        if is_classification:
            inferred = self._prediction_to_labels(inferred, is_regression=False)
            target = self._normalize_ground_truth(target, is_regression=False)
        else:
            inferred = self._prediction_to_labels(inferred, is_regression=True)
            target = self._normalize_ground_truth(target, is_regression=True)
        self.attack_prediction_time = end_time - start_time
        sensitive_attribute = _sensitive_slice(
            getattr(data, "_sensitive_train", None),
            self.attack_size,
        )
        score_dict = self._score(
            attack_kind="attribute",
            y_true=target,
            y_pred=inferred,
            targeted_attribute=targeted_attribute_string,
            is_classification=is_classification,
            attack_generation_time=self.attack_time,
            data=data,
            sensitive_features=sensitive_attribute,
        )
        return self._finalize_attack_state(
            attack=inferred,
            attack_predictions=inferred,
            attacked_labels=target,
            score_dict=score_dict,
            score_y_pred=inferred,
            score_y_proba=None,
        )

    def infer_membership(self, data: DataConfig, attack: AttackLike) -> ScoreDict:
        """Infer whether sampled records belonged to the model training set.

        Args:
            data: Runtime data config providing train/test splits.
            attack: Instantiated membership-inference attack implementation.

        Returns:
            Score payload for inferred membership labels.
        """
        start_time = time.perf_counter()
        y_train_raw = self._prepare_labels_for_attack(getattr(data, "y_train"))
        y_train_values = self._to_numpy_array(y_train_raw)
        if y_train_values.ndim == 1:
            y_data = pd.get_dummies(y_train_values).values
        elif y_train_values.ndim == 2 and y_train_values.shape[1] == 1:
            y_data = pd.get_dummies(y_train_values.ravel()).values
        else:
            y_data = y_train_values
        try:
            attack.fit(
                x=self._prepare_features_for_art(getattr(data, "X_train")),
                y=y_data,
                test_x=self._prepare_features_for_art(getattr(data, "X_test")),
            )
        except AxisError:
            safe_y_data = pd.get_dummies(
                np.asarray(y_train_values).reshape(-1),
            ).values
            attack.fit(
                x=self._prepare_features_for_art(getattr(data, "X_train")),
                y=safe_y_data,
                test_x=self._prepare_features_for_art(getattr(data, "X_test")),
            )
        self.attack_time = time.perf_counter() - start_time

        x_train = self._prepare_features_for_art(getattr(data, "X_train"))
        x_test = self._prepare_features_for_art(getattr(data, "X_test"))
        big_X = np.vstack(
            (
                self._to_numpy_array(x_train),
                self._to_numpy_array(x_test),
            ),
        )
        big_y = np.hstack(
            (
                self._to_numpy_array(
                    self._prepare_labels_for_attack(getattr(data, "y_train")),
                ),
                self._to_numpy_array(
                    self._prepare_labels_for_attack(getattr(data, "y_test")),
                ),
            ),
        )
        labels = np.array([1] * len(data.X_train) + [0] * len(data.X_test))
        sensitive_train = getattr(data, "_sensitive_train", None)
        sensitive_test = getattr(data, "_sensitive_test", None)
        if sensitive_train is not None and sensitive_test is not None:
            big_sensitive = np.concatenate(
                [np.asarray(sensitive_train), np.asarray(sensitive_test)],
            )
        else:
            big_sensitive = None
        n = self.attack_size
        indices = np.arange(len(big_X))
        indices = np.random.choice(indices, size=n, replace=False)
        big_X = big_X[indices]
        big_y = big_y[indices]
        labels = labels[indices]
        sensitive_membership = (
            big_sensitive[indices] if big_sensitive is not None else None
        )
        start_time = time.perf_counter()
        inferred = attack.infer(
            x=big_X,
            y=big_y,
        )
        end_time = time.perf_counter()
        self.attack_time = end_time - start_time
        inferred = self._normalize_inferred_output(inferred)
        assert (
            len(inferred) == n
        ), f"Length of inferred {len(inferred)} does not match number of samples {self.attack_size}"
        start_time = time.perf_counter()
        inferred = self._normalize_inferred_output(inferred, reference=labels)
        inferred = self._prediction_to_labels(inferred, is_regression=False)
        labels = self._normalize_ground_truth(labels, is_regression=False)
        end_time = time.perf_counter()
        self.attack_prediction_time = end_time - start_time
        score_dict = self._score(
            attack_kind="membership",
            y_true=labels,
            y_pred=inferred,
            data=data,
            sensitive_features=sensitive_membership,
        )
        return self._finalize_attack_state(
            attack=inferred,
            attack_predictions=inferred,
            attacked_labels=labels,
            score_dict=score_dict,
            score_y_pred=inferred,
            score_y_proba=None,
        )

    def _resolve_source_split(
        self,
        data,
        *,
        attack_kind: str,
    ) -> tuple[str, object, object]:
        requested_mode = self.resolve_mode_for_attack_kind(
            attack_kind,
            attack_sub_family=self.attack_sub_family,
        )
        return resolve_attack_split_payload(
            data,
            requested_mode,
            error_message=(
                "Inference attacks require test features/labels (or val when mode='val')."
            ),
            on_fallback=lambda mode: logger.warning(
                "Attack mode='%s' requested but %s split is unavailable; falling back to test split.",
                mode,
                "validation" if mode == "val" else "training",
            ),
        )

    def infer_model_inversion(self, data: DataConfig, attack: AttackLike) -> ScoreDict:
        """Reconstruct representative inputs for target class labels.

        Args:
            data: Runtime data config providing source samples and labels.
            attack: Instantiated model-inversion attack implementation.

        Returns:
            Score payload comparing reconstructed inputs against class prototypes.

        Raises:
            ValueError: If source split, targets, or initialization values are invalid.
        """
        split, x_source, y_source = self._resolve_source_split(
            data,
            attack_kind="model_inversion",
        )
        x_source = self._to_numpy_array(
            self._prepare_features_for_attack(x_source),
            dtype=ART_NUMPY_DTYPE,
        )
        y_source = self._normalize_ground_truth(y_source, is_regression=False)

        if len(x_source) == 0:
            raise ValueError("Model inversion requires at least one source sample.")

        targets_param = self.attack_params.get("targets", None)
        if targets_param is None:
            target_labels = np.unique(y_source.astype(int))
        else:
            target_labels = np.asarray(targets_param).reshape(-1).astype(int)

        if len(target_labels) == 0:
            raise ValueError("Model inversion requires at least one target label.")

        if self.attack_size is not None and int(self.attack_size) > 0:
            target_labels = target_labels[: int(self.attack_size)]

        init_samples = self.attack_params.get("x_init", None)
        if init_samples is None:
            init_mode = str(
                self.attack_params.get("initialization", "average"),
            ).lower()
            sample_shape = tuple(x_source.shape[1:])
            if init_mode == "zeros":
                init_samples = np.zeros(
                    (len(target_labels),) + sample_shape,
                    dtype=ART_NUMPY_DTYPE,
                )
            elif init_mode == "ones":
                init_samples = np.ones(
                    (len(target_labels),) + sample_shape,
                    dtype=ART_NUMPY_DTYPE,
                )
            elif init_mode == "random":
                init_samples = np.random.uniform(
                    low=0.0,
                    high=1.0,
                    size=(len(target_labels),) + sample_shape,
                ).astype(ART_NUMPY_DTYPE)
            elif init_mode == "average":
                avg = np.mean(x_source, axis=0)
                init_samples = np.repeat(
                    avg[None, ...],
                    repeats=len(target_labels),
                    axis=0,
                ).astype(ART_NUMPY_DTYPE)
            else:
                raise ValueError(
                    "Unsupported model inversion initialization "
                    f"'{init_mode}'. Use one of: zeros, ones, random, average.",
                )
        else:
            init_samples = self._to_numpy_array(init_samples, dtype=ART_NUMPY_DTYPE)

        if len(init_samples) != len(target_labels):
            raise ValueError(
                "Length mismatch between model inversion initial samples and targets: "
                f"len(x_init)={len(init_samples)} len(targets)={len(target_labels)}",
            )

        start_time = time.perf_counter()
        try:
            inferred = attack.infer(x=init_samples, y=target_labels)
        except TypeError:
            inferred = attack.infer(init_samples, target_labels)
        self.attack_time = time.perf_counter() - start_time

        self.attack_prediction_time = 0.0

        start_time = time.perf_counter()
        inferred_arr = self._to_numpy_array(inferred, dtype=ART_NUMPY_DTYPE)
        inferred_flat = inferred_arr.reshape(len(target_labels), -1)

        fallback_proto = np.mean(x_source, axis=0).reshape(-1)
        prototypes = []
        for label in target_labels:
            class_mask = y_source.astype(int) == int(label)
            if np.any(class_mask):
                class_mean = np.mean(x_source[class_mask], axis=0).reshape(-1)
                prototypes.append(class_mean)
            else:
                prototypes.append(fallback_proto)
        proto_arr = np.asarray(prototypes, dtype=ART_NUMPY_DTYPE)

        score_dict = self._score(
            attack_kind="attribute",
            y_true=proto_arr,
            y_pred=inferred_flat,
            targeted_attribute="model_inversion",
            is_classification=False,
            attack_generation_time=self.attack_time,
            data=data,
        )
        self.attack_score_time = float(score_dict.get("attack_score_time", 0.0))

        model_inversion_scores = {
            "model_inversion_mse": score_dict.get("inferred_model_inversion_mse"),
            "model_inversion_mae": score_dict.get("inferred_model_inversion_mae"),
            "model_inversion_num_targets": int(len(target_labels)),
            "model_inversion_mode": split,
        }
        return self._finalize_attack_state(
            attack=inferred_arr,
            attack_predictions=inferred_arr,
            attacked_labels=target_labels,
            score_dict={**score_dict, **model_inversion_scores},
        )

    def infer_database_reconstruction(
        self,
        data: DataConfig,
        attack: AttackLike,
    ) -> ScoreDict:
        """Reconstruct a held-out database row from the remaining dataset.

        Args:
            data: Runtime data config providing the source split to reconstruct from.
            attack: Instantiated reconstruction attack implementation.

        Returns:
            Score payload for the reconstructed record and auxiliary metadata.

        Raises:
            ValueError: If source split is too small or reconstruction indices are invalid.
        """
        split, x_source, y_source_raw = self._resolve_source_split(
            data,
            attack_kind="reconstruction",
        )
        x_source = self._to_numpy_array(
            self._prepare_features_for_attack(x_source),
            dtype=ART_NUMPY_DTYPE,
        )

        if len(x_source) < 2:
            raise ValueError(
                "Database reconstruction requires at least two rows in the selected split.",
            )

        missing_index = int(self.attack_params.get("missing_index", -1))
        if missing_index < 0:
            missing_index = len(x_source) + missing_index
        if missing_index < 0 or missing_index >= len(x_source):
            raise ValueError(
                "database reconstruction missing_index is out of bounds: "
                f"{missing_index} for split size {len(x_source)}",
            )

        x_true_missing = x_source[missing_index : missing_index + 1]
        x_known = np.delete(x_source, missing_index, axis=0)

        y_known = None
        y_true_missing = None
        if y_source_raw is not None:
            y_source = self._to_numpy_array(
                self._prepare_labels_for_attack(y_source_raw),
            )
            y_true_missing = y_source.reshape(-1)[missing_index]
            y_known = np.delete(y_source, missing_index, axis=0)

        start_time = time.perf_counter()
        try:
            reconstructed = attack.reconstruct(x_known, y_known)
        except TypeError:
            reconstructed = attack.reconstruct(x_known)
        self.attack_time = time.perf_counter() - start_time

        self.attack_prediction_time = 0.0

        start_time = time.perf_counter()
        if isinstance(reconstructed, tuple):
            if len(reconstructed) == 0:
                raise ValueError("DatabaseReconstruction returned an empty tuple.")
            x_reconstructed = reconstructed[0]
            y_reconstructed = reconstructed[1] if len(reconstructed) > 1 else None
        else:
            x_reconstructed = reconstructed
            y_reconstructed = None

        x_reconstructed = self._to_numpy_array(x_reconstructed, dtype=ART_NUMPY_DTYPE)
        if x_reconstructed.ndim == 1:
            x_reconstructed = x_reconstructed.reshape(1, -1)
        x_pred = x_reconstructed.reshape(x_reconstructed.shape[0], -1)
        x_true = x_true_missing.reshape(1, -1)

        x_pred_row = x_pred[:1]

        feature_scores = self._score(
            attack_kind="attribute",
            y_true=x_true,
            y_pred=x_pred_row,
            targeted_attribute="database_reconstruction_feature",
            is_classification=False,
            attack_generation_time=self.attack_time,
            data=data,
        )

        label_score = {}
        if y_reconstructed is not None and y_true_missing is not None:
            y_pred = self._to_numpy_array(y_reconstructed).reshape(-1)
            if len(y_pred) > 0:
                task_is_classification = bool(
                    self._infer_task_is_classification(data, None),
                )
                y_pred_first = y_pred[0]
                if task_is_classification:
                    raw_label_scores = self._score(
                        attack_kind="attribute",
                        y_true=np.asarray([int(y_true_missing)]),
                        y_pred=np.asarray([int(y_pred_first)]),
                        targeted_attribute="database_reconstruction_label",
                        is_classification=True,
                        data=data,
                    )
                    label_score = {
                        "database_reconstruction_label_accuracy": raw_label_scores.get(
                            "inferred_database_reconstruction_label_accuracy",
                        ),
                    }
                else:
                    raw_label_scores = self._score(
                        attack_kind="attribute",
                        y_true=np.asarray([float(y_true_missing)]),
                        y_pred=np.asarray([float(y_pred_first)]),
                        targeted_attribute="database_reconstruction_label",
                        is_classification=False,
                        data=data,
                    )
                    label_score = {
                        "database_reconstruction_label_mae": raw_label_scores.get(
                            "inferred_database_reconstruction_label_mae",
                        ),
                    }
                    feature_scores = ScoreDict.from_payload(
                        {**feature_scores, **raw_label_scores},
                    )

        self.attack_score_time = time.perf_counter() - start_time

        compatibility_scores = {
            "database_reconstruction_feature_mse": feature_scores.get(
                "inferred_database_reconstruction_feature_mse",
            ),
            "database_reconstruction_feature_mae": feature_scores.get(
                "inferred_database_reconstruction_feature_mae",
            ),
        }
        return self._finalize_attack_state(
            attack=x_reconstructed,
            attack_predictions=x_reconstructed,
            attacked_labels=x_true_missing,
            score_dict={
                **feature_scores,
                **compatibility_scores,
                "database_reconstruction_num_features": int(x_true.shape[1]),
                "database_reconstruction_num_known_rows": int(len(x_known)),
                "database_reconstruction_missing_index": int(missing_index),
                **label_score,
                "attack_size": int(x_pred.shape[0]),
                "attack_score_time": float(self.attack_score_time),
            },
        )
