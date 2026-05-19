"""Configuration for regularizer defenses (training-time regularization)."""

from dataclasses import dataclass, field
from typing import Any

from deckard.plugins.defense import DefenseTypePlugin

from ..utils import safe_store
from .defend import DefensePipelineConfig, _DefenseMixin


class _RegularizerDefenseMixin(_DefenseMixin):
    """Reusable regularizer defense behavior."""

    def __call__(
        self,
        *,
        data,
        defense_type,
        defense_subtype,
        defense_class,
        art_class,
        init_params,
        base_estimator,
        existing_preprocessors,
        existing_postprocessors,
    ) -> tuple[Any, Any]:
        raise NotImplementedError(
            "Regularizer defenses are not implemented yet.",
        )


@dataclass(eq=False, kw_only=True)
class RegularizerDefenseConfig(_RegularizerDefenseMixin, DefensePipelineConfig):
    """Configuration for regularizer-based defenses.

    Registers regularizer defense behavior and plugin metadata used during
    defense runtime dispatch.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=_RegularizerDefenseMixin,
                defense_type="regularizer",
            ),
        ],
    )


# Register regularizer defense config
safe_store(
    group="model",
    name="regularizer_defense",
    node=RegularizerDefenseConfig(),
)

safe_store(
    group="search/models",
    name="regularizer_defense",
    node=RegularizerDefenseConfig(),
)
