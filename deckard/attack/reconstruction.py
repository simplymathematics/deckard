"""Configuration for reconstruction attacks (database reconstruction)."""

from dataclasses import dataclass
from sklearn.base import BaseEstimator

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..frameworks.types import AttackLike, EstimatorLike
from ..model import ModelConfig
from .base import AttackFamily, AttackSubFamily
from .inference import InferenceAttackConfig


@dataclass(eq=False, kw_only=True)
class ReconstructionAttackConfig(InferenceAttackConfig):
    """Configuration for database reconstruction attacks.

    Attributes:
        score_dict: Runtime score payload for reconstruction metrics.
    """

    def reconstruct(
        self,
        data: DataConfig,
        attack: AttackLike,
    ) -> ScoreDict:
        """Execute model reconstruction attack with normalized scoring output.

        Args:
            data: Runtime dataset and split container.
            attack: Instantiated reconstruction attack implementation.

        Returns:
            Score payload for reconstruction attack execution.
        """
        return self.infer_database_reconstruction(data=data, attack=attack)

    def infer_database_reconstruction(
        self,
        data: DataConfig,
        attack: AttackLike,
    ) -> ScoreDict:
        """Execute database reconstruction inference for the provided attack.

        Args:
            data: Runtime dataset and split container.
            attack: Instantiated reconstruction attack implementation.

        Returns:
            Score payload for reconstructed database-row inference.
        """
        return super().infer_database_reconstruction(data=data, attack=attack)

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
        """Dispatch reconstruction inference attack execution for matching subtype.

        Args:
            data: Runtime dataset and split container.
            model: User model configuration or estimator.
            art_model: ART-wrapped model used by reconstruction attack.
            attack: Instantiated reconstruction attack implementation.
            attack_family: Parsed attack family.
            attack_sub_family: Parsed attack sub-family.

        Returns:
            Score payload for reconstruction attack execution.

        Raises:
            ValueError: If attack family/subtype is not inference.reconstruction.
        """
        _ = (model, art_model)
        if (attack_family or "").lower() != "inference" or (
            attack_sub_family or ""
        ).lower() != "reconstruction":
            raise ValueError(
                "_ReconstructionAttackConfig requires inference.reconstruction attack subtype",
            )
        return self.reconstruct(data=data, attack=attack)

    # Note:
    #     Expected family/subtype is ``inference.reconstruction``.
