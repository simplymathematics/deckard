"""Configuration for reconstruction attacks (database reconstruction)."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from .base import AttackConfig, AttackTypePlugin
from .inference import _InferenceAttackMixin
from ..utils import safe_store




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
    """Configuration for database reconstruction attacks.

    Initialization params
    ---------------------
    attack_type : str
        Attack family path inherited from ``AttackConfig``. Expected family is
        ``inference`` for reconstruction.
    attack_params : dict[str, Any]
        Constructor kwargs and runtime controls used by reconstruction attacks.
    init_params : dict[str, Any]
        Metadata-only declaration payload for class/type/library docs.
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
            )
        ]
    )


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
