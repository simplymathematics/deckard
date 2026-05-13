"""Configuration for reconstruction attacks (database reconstruction)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from .base import AttackConfig
from .inference import _InferenceAttackMixin
from ..utils import safe_store

if TYPE_CHECKING:
    pass


class _ReconstructionAttackMixin(_InferenceAttackMixin):
    """Reusable database reconstruction attack behavior."""

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
        return self._infer_database_reconstruction(data=data, attack=attack)


@dataclass(eq=False)
class ReconstructionAttackConfig(_ReconstructionAttackMixin, AttackConfig):
    """
    Configuration for database reconstruction attacks.
    
    Reconstruction attacks attempt to infer or recover the entire training
    database from a model's predictions. Used to assess privacy leakage
    from machine learning models.
    """

    pass


# Register reconstruction attack config
safe_store(
    group="attack",
    name="reconstruction",
    node=ReconstructionAttackConfig(),
)

safe_store(
    group="search/attack",
    name="reconstruction",
    node=ReconstructionAttackConfig(),
)
