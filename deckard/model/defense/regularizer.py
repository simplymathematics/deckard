"""Configuration for regularizer defenses (training-time regularization)."""

from dataclasses import dataclass, field
from typing import Any

from deckard.plugins.defense import DefenseTypePlugin

from ...utils import safe_store
from .base import DefensePipelineConfig, DefenseMixin


class RegularizerDefenseMixin(DefenseMixin):
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
        """Raise for unimplemented regularizer defense runtime path."""
        raise NotImplementedError(
            "Regularizer defenses are not implemented yet.",
        )


@dataclass(eq=False, kw_only=True)
class RegularizerDefenseConfig(RegularizerDefenseMixin, DefensePipelineConfig):
    """Configuration for regularizer-based defenses.

    Registers regularizer defense behavior and plugin metadata used during
    defense runtime dispatch.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=RegularizerDefenseMixin,
                defense_type="regularizer",
            ),
        ],
    )


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
