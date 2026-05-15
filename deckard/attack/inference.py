"""Configuration for inference attacks (membership, attribute, model inversion)."""

import time
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from art.config import ART_NUMPY_DTYPE
from omegaconf import ListConfig, OmegaConf
from numpy.exceptions import AxisError

from .base import AttackConfig, AttackTypePlugin, _AttackMixin, _sensitive_slice



logger = logging.getLogger(__name__)


class _InferenceAttackMixin(_AttackMixin):
    """Reusable inference attack behavior (membership, attribute, inversion)."""

    targeted_attribute: str

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
        if (attack_type or "").lower() != "inference":
            raise ValueError(
                f"_InferenceAttackMixin received unsupported attack type: {attack_type}",
            )
        subtype = (attack_subtype or "").lower()
        if subtype == "membership_inference":
            return self._infer_membership(data=data, attack=attack)
        if subtype == "attribute_inference":
            assert (
                self.targeted_attribute is not None
            ), "targeted_attribute must be specified for inference attacks"
            return self._infer_attribute(
                data=data,
                art_model=art_model,
                attack=attack,
                targeted_attribute=self.targeted_attribute,
            )
        if subtype == "model_inversion":
            return self._infer_model_inversion(data=data, attack=attack)
        if subtype == "reconstruction":
            return self._infer_database_reconstruction(data=data, attack=attack)
        raise ValueError(f"Unsupported inference attack subtype: {attack_subtype}")

    def _infer_attribute(
        self,
        data,
        art_model,
        attack,
        targeted_attribute,
    ) -> dict:
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
        active_mode = self.resolve_mode_for_attack_kind("attribute")
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
        start_time = time.process_time()
        attack.fit(x=X_test_subset)
        attack_time = time.process_time() - start_time
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
        start_time = time.process_time()
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
        end_time = time.process_time()
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
            sensitive_features=sensitive_attribute,
        )
        self.score_y_pred = inferred
        self.score_y_proba = None
        self.score_dict = {**self.score_dict, **score_dict}
        self.attack = inferred
        self.attack_predictions = inferred
        self.attacked_labels = target
        return self.score_dict

    def _infer_membership(self, data, attack) -> dict:
        start_time = time.process_time()
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
        self.attack_time = time.process_time() - start_time

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
        start_time = time.process_time()
        inferred = attack.infer(
            x=big_X,
            y=big_y,
        )
        end_time = time.process_time()
        self.attack_time = end_time - start_time
        inferred = self._normalize_inferred_output(inferred)
        assert (
            len(inferred) == n
        ), f"Length of inferred {len(inferred)} does not match number of samples {self.attack_size}"
        start_time = time.process_time()
        inferred = self._normalize_inferred_output(inferred, reference=labels)
        inferred = self._prediction_to_labels(inferred, is_regression=False)
        labels = self._normalize_ground_truth(labels, is_regression=False)
        self.attack_predictions = inferred
        self.attacked_labels = labels
        end_time = time.process_time()
        self.attack_prediction_time = end_time - start_time
        score_dict = self._score(
            attack_kind="membership",
            y_true=labels,
            y_pred=inferred,
            sensitive_features=sensitive_membership,
        )
        self.score_y_pred = inferred
        self.score_y_proba = None
        self.score_dict = {**self.score_dict, **score_dict}
        self.attack = inferred
        return self.score_dict

    def _infer_model_inversion(self, data, attack) -> dict:
        split = str(self.attack_params.get("split", "test")).lower()
        if split not in {"train", "test"}:
            raise ValueError(
                f"Unsupported model inversion split '{split}'. Expected 'train' or 'test'.",
            )

        x_source = getattr(data, "X_train" if split == "train" else "X_test")
        y_source = getattr(data, "y_train" if split == "train" else "y_test")
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

        start_time = time.process_time()
        try:
            inferred = attack.infer(x=init_samples, y=target_labels)
        except TypeError:
            inferred = attack.infer(init_samples, target_labels)
        self.attack_time = time.process_time() - start_time

        self.attack_prediction_time = 0.0

        start_time = time.process_time()
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

        mse = float(np.mean((inferred_flat - proto_arr) ** 2))
        mae = float(np.mean(np.abs(inferred_flat - proto_arr)))
        self.attack_score_time = time.process_time() - start_time

        self.attack_predictions = inferred_arr
        self.attacked_labels = target_labels
        self.attack = inferred_arr

        self.score_dict = {
            **self.score_dict,
            "model_inversion_mse": mse,
            "model_inversion_mae": mae,
            "model_inversion_num_targets": int(len(target_labels)),
            "attack_size": int(len(target_labels)),
            "attack_score_time": float(self.attack_score_time),
        }
        return self.score_dict

    def _infer_database_reconstruction(self, data, attack) -> dict:
        split = str(self.attack_params.get("split", "train")).lower()
        if split not in {"train", "test"}:
            raise ValueError(
                "Unsupported database reconstruction split "
                f"'{split}'. Expected 'train' or 'test'.",
            )

        x_source = getattr(data, "X_train" if split == "train" else "X_test")
        y_source_raw = getattr(data, "y_train" if split == "train" else "y_test", None)
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

        start_time = time.process_time()
        try:
            reconstructed = attack.reconstruct(x_known, y_known)
        except TypeError:
            reconstructed = attack.reconstruct(x_known)
        self.attack_time = time.process_time() - start_time

        self.attack_prediction_time = 0.0

        start_time = time.process_time()
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
        feature_mse = float(np.mean((x_pred_row - x_true) ** 2))
        feature_mae = float(np.mean(np.abs(x_pred_row - x_true)))

        label_score = {}
        if y_reconstructed is not None and y_true_missing is not None:
            y_pred = self._to_numpy_array(y_reconstructed).reshape(-1)
            if len(y_pred) > 0:
                task_is_classification = bool(
                    self._infer_task_is_classification(data, None),
                )
                y_pred_first = y_pred[0]
                if task_is_classification:
                    label_score = {
                        "database_reconstruction_label_accuracy": float(
                            int(y_pred_first) == int(y_true_missing),
                        ),
                    }
                else:
                    label_score = {
                        "database_reconstruction_label_mae": float(
                            np.abs(float(y_pred_first) - float(y_true_missing)),
                        ),
                    }

        self.attack_score_time = time.process_time() - start_time

        self.attack_predictions = x_reconstructed
        self.attacked_labels = x_true_missing
        self.attack = x_reconstructed

        self.score_dict = {
            **self.score_dict,
            "database_reconstruction_feature_mse": feature_mse,
            "database_reconstruction_feature_mae": feature_mae,
            "database_reconstruction_num_features": int(x_true.shape[1]),
            "database_reconstruction_num_known_rows": int(len(x_known)),
            "database_reconstruction_missing_index": int(missing_index),
            **label_score,
            "attack_size": int(x_pred.shape[0]),
            "attack_score_time": float(self.attack_score_time),
        }
        return self.score_dict


@dataclass(eq=False, kw_only=True)
class InferenceAttackConfig(_InferenceAttackMixin, AttackConfig):
    """Configuration for privacy inference attacks.

    Initialization params
    ---------------------
    attack_type : str
        Attack family path inherited from ``AttackConfig``. Expected family is
        ``inference``.
    attack_params : dict[str, Any]
        Constructor kwargs and runtime controls for membership, attribute,
        model-inversion, and related inference subtypes.
    init_params : dict[str, Any]
        Metadata-only declaration payload for class/type/library docs.
    plugins : list[AttackTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``AttackTypePlugin`` configured with:
        ``mixin_type: type = _InferenceAttackMixin``,
        ``attack_type: str = 'inference'``, and
        ``excluded_subtypes: tuple[str, ...] = ('reconstruction',)``.

    Runtime params
    --------------
    _InferenceAttackMixin.__call__(self, *, data: Any, model: Any, art_model: Any, attack: Any, attack_type: str, attack_subtype: str) -> dict
        Runtime dispatch entrypoint invoked by ``AttackConfig.__call__``.
    _InferenceAttackMixin._infer_membership(self, data: Any, attack: Any) -> dict
        Membership-inference runtime handler.
    _InferenceAttackMixin._infer_attribute(self, data: Any, art_model: Any, attack: Any, targeted_attribute: str | int) -> dict
        Attribute-inference runtime handler.
    _InferenceAttackMixin._infer_model_inversion(self, data: Any, attack: Any) -> dict
        Model-inversion runtime handler.
    """

    plugins: list = field(
        default_factory=lambda: [
            AttackTypePlugin(
                mixin_type=_InferenceAttackMixin,
                attack_type="inference",
                excluded_subtypes=("reconstruction",),
            )
        ]
    )


