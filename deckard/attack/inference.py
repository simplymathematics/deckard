"""Configuration for inference attacks (membership, attribute, model inversion)."""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from art.config import ART_NUMPY_DTYPE
from numpy.exceptions import AxisError
from omegaconf import ListConfig, OmegaConf

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..model import ModelConfig
from ..frameworks.types import AttackLike, EstimatorLike, StringifiedClass
from .base import AttackConfig, AttackTypePlugin, AttackMixin, _sensitive_slice

logger = logging.getLogger(__name__)


class InferenceAttackMixin(AttackMixin):
    """Reusable inference attack behavior (membership, attribute, inversion)."""

    targeted_attribute: str

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
        """Dispatch inference attack execution for supported inference subtypes.

        Args:
            data: Runtime dataset and split container.
            model: User model configuration or estimator.
            art_model: ART-wrapped model used for inference attacks.
            attack: Instantiated inference attack implementation.
            attack_type: Parsed attack family.
            attack_subtype: Parsed attack subtype.

        Returns:
            Score payload for the selected inference subtype.

        Raises:
            ValueError: If attack family/subtype is unsupported.
        """
        if (attack_type or "").lower() != "inference":
            raise ValueError(
                f"_InferenceAttackMixin received unsupported attack type: {attack_type}",
            )
        subtype = (attack_subtype or "").lower()
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
        raise ValueError(f"Unsupported inference attack subtype: {attack_subtype}")

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
            attack_subtype=self.attack_subtype,
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
            sensitive_features=sensitive_attribute,
        )
        self.score_y_pred = inferred
        self.score_y_proba = None
        self.score_dict = ScoreDict.from_payload({**self.score_dict, **score_dict})
        self.attack = inferred
        self.attack_predictions = inferred
        self.attacked_labels = target
        return ScoreDict.from_payload(self.score_dict)

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
        self.attack_predictions = inferred
        self.attacked_labels = labels
        end_time = time.perf_counter()
        self.attack_prediction_time = end_time - start_time
        score_dict = self._score(
            attack_kind="membership",
            y_true=labels,
            y_pred=inferred,
            sensitive_features=sensitive_membership,
        )
        self.score_y_pred = inferred
        self.score_y_proba = None
        self.score_dict = ScoreDict.from_payload({**self.score_dict, **score_dict})
        self.attack = inferred
        return ScoreDict.from_payload(self.score_dict)

    def _resolve_source_split(
        self,
        data,
        *,
        attack_kind: str,
    ) -> tuple[str, object, object]:
        requested_mode = self.resolve_mode_for_attack_kind(
            attack_kind,
            attack_subtype=self.attack_subtype,
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
                "Inference attacks require test features/labels (or val when mode='val').",
            )
        return "test", X_test, y_test

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
        )
        self.attack_score_time = float(score_dict.get("attack_score_time", 0.0))

        self.attack_predictions = inferred_arr
        self.attacked_labels = target_labels
        self.attack = inferred_arr

        model_inversion_scores = {
            "model_inversion_mse": score_dict.get("inferred_model_inversion_mse"),
            "model_inversion_mae": score_dict.get("inferred_model_inversion_mae"),
            "model_inversion_num_targets": int(len(target_labels)),
            "model_inversion_mode": split,
        }
        self.score_dict = ScoreDict.from_payload(
            {
                **self.score_dict,
                **score_dict,
                **model_inversion_scores,
            },
        )
        return ScoreDict.from_payload(self.score_dict)

    def infer_database_reconstruction(self, data: DataConfig, attack: AttackLike) -> ScoreDict:
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

        self.attack_predictions = x_reconstructed
        self.attacked_labels = x_true_missing
        self.attack = x_reconstructed

        compatibility_scores = {
            "database_reconstruction_feature_mse": feature_scores.get(
                "inferred_database_reconstruction_feature_mse",
            ),
            "database_reconstruction_feature_mae": feature_scores.get(
                "inferred_database_reconstruction_feature_mae",
            ),
        }
        self.score_dict = ScoreDict.from_payload(
            {
                **self.score_dict,
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
        return ScoreDict.from_payload(self.score_dict)


@dataclass(eq=False, kw_only=True)
class InferenceAttackConfig(InferenceAttackMixin, AttackConfig):
    """Configuration for privacy inference attacks.

    Initialization params
    ---------------------
    attack_type : str
        Attack family path inherited from ``AttackConfig``. Expected family is
        ``inference``.
    attack_params : dict[str, Any]
        Constructor kwargs and runtime controls for membership, attribute,
        model-inversion, and related inference subtypes.
    plugins : list[AttackTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``AttackTypePlugin`` configured with:
        ``mixin_type: type = _InferenceAttackMixin``,
        ``attack_type: str = 'inference'``, and
        ``excluded_subtypes: tuple[str, ...] = ('reconstruction',)``.

    Runtime params
    --------------
    _InferenceAttackMixin.__call__(self, *, data: Any, model: Any, art_model: Any, attack: Any, attack_type: str, attack_subtype: str) -> ScoreDict
        Runtime dispatch entrypoint invoked by ``AttackConfig.__call__``.
    _InferenceAttackMixin.infer_membership(self, data: Any, attack: Any) -> ScoreDict
        Membership-inference runtime handler.
    _InferenceAttackMixin.infer_attribute(self, data: Any, art_model: Any, attack: Any, targeted_attribute: str | int) -> ScoreDict
        Attribute-inference runtime handler.
    _InferenceAttackMixin.infer_model_inversion(self, data: Any, attack: Any) -> ScoreDict
        Model-inversion runtime handler.
    """

    plugins: list = field(
        default_factory=lambda: [
            AttackTypePlugin(
                mixin_type=InferenceAttackMixin,
                attack_type="inference",
                excluded_subtypes=("reconstruction",),
            ),
        ],
    )
