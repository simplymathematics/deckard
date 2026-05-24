"""Configuration for reconstruction attacks (database reconstruction)."""

from dataclasses import dataclass, field
from sklearn.base import BaseEstimator

from ..artifacts import ScoreDict
from ..data import DataConfig
from ..frameworks.types import AttackLike, EstimatorLike, StringifiedClass
from ..model import ModelConfig
from .base import AttackConfig, AttackTypePlugin
from .inference import InferenceAttackMixin


class ReconstructionAttackMixin(InferenceAttackMixin):
    """Reusable database reconstruction attack behavior."""

    def reconstruct(
        self,
        data: DataConfig,
        attack: AttackLike,
    ) -> ScoreDict:
        """Public subtype-mirroring alias for reconstruction execution."""
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
        attack_type: StringifiedClass,
        attack_subtype: StringifiedClass,
    ) -> ScoreDict:
        """Dispatch reconstruction inference attack execution for matching subtype.

        Args:
            data: Runtime dataset and split container.
            model: User model configuration or estimator.
            art_model: ART-wrapped model used by reconstruction attack.
            attack: Instantiated reconstruction attack implementation.
            attack_type: Parsed attack family.
            attack_subtype: Parsed attack subtype.

        Returns:
            Score payload for reconstruction attack execution.

        Raises:
            ValueError: If attack family/subtype is not inference.reconstruction.
        """
        if (attack_type or "").lower() != "inference" or (
            attack_subtype or ""
        ).lower() != "reconstruction":
            raise ValueError(
                "_ReconstructionAttackMixin requires inference.reconstruction attack subtype",
            )
        return self.reconstruct(data=data, attack=attack)


@dataclass(eq=False, kw_only=True)
class ReconstructionAttackConfig(ReconstructionAttackMixin, AttackConfig):
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
    _ReconstructionAttackMixin.__call__(self, *, data: Any, model: Any, art_model: Any, attack: Any, attack_type: str, attack_subtype: str) -> ScoreDict
        Runtime dispatch entrypoint for ``inference.reconstruction`` subtype.
    """

    plugins: list = field(
        default_factory=lambda: [
            AttackTypePlugin(
                mixin_type=ReconstructionAttackMixin,
                attack_type="inference",
                attack_subtype="reconstruction",
            ),
        ],
    )
