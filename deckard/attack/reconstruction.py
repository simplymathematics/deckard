"""Configuration for reconstruction attacks (database reconstruction)."""

from dataclasses import dataclass, field

from .base import AttackConfig, AttackTypePlugin
from .inference import _InferenceAttackMixin


class _ReconstructionAttackMixin(_InferenceAttackMixin):
    """Reusable database reconstruction attack behavior."""
    def infer_database_reconstruction(self, data, attack) -> dict:
        split = str(self.attack_params.get("split", "train")).lower()
        if split not in {"train", "test", "val"}:
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
        if (attack_type or "").lower() != "inference" or (
            attack_subtype or ""
        ).lower() != "reconstruction":
            raise ValueError(
                "_ReconstructionAttackMixin requires inference.reconstruction attack subtype",
            )
        return self.infer_database_reconstruction(data=data, attack=attack)


@dataclass(eq=False, kw_only=True)
class ReconstructionAttackConfig(_ReconstructionAttackMixin, AttackConfig):
    """Configuration for database reconstruction attacks.

    Initialization params
    ---------------------
    attack_type : str
        Attack family path inherited from ``AttackConfig``. Expected family is
        ``inference`` for reconstruction.
    attack_params : dict[str, Any]
        Constructor kwargs and runtime controls used by reconstruction attacks.
    plugins : list[AttackTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``AttackTypePlugin`` configured with:
        ``mixin_type: type = _ReconstructionAttackMixin``,
        ``attack_type: str = 'inference'``, and
        ``attack_subtype: str = 'reconstruction'``.

    Runtime params
    --------------
    _ReconstructionAttackMixin.__call__(self, *, data: Any, model: Any, art_model: Any, attack: Any, attack_type: str, attack_subtype: str) -> dict
        Runtime dispatch entrypoint for ``inference.reconstruction`` subtype.
    """

    plugins: list = field(
        default_factory=lambda: [
            AttackTypePlugin(
                mixin_type=_ReconstructionAttackMixin,
                attack_type="inference",
                attack_subtype="reconstruction",
            ),
        ],
    )
