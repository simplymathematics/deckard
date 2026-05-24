"""Configuration for postprocessor defenses (output transformation)."""

from dataclasses import dataclass, field
from typing import Any

from deckard.plugins.defense import DefenseTypePlugin

from ...utils import safe_store
from .base import DefensePipelineConfig, DefenseMixin


class PostprocessorDefenseMixin(DefenseMixin):
    """Reusable postprocessor defense behavior."""

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
        """Attach postprocessor defense and return defense with defended estimator."""
        assert defense_class is not None
        defense = defense_class(**(self.defense_params or {}))
        defended_estimator = self._build_art_wrapper(
            art_class=art_class,
            base_estimator=base_estimator,
            init_params=init_params,
            preprocessing_defences=existing_preprocessors,
            postprocessing_defences=existing_postprocessors + [defense],
        )
        return defense, defended_estimator


@dataclass(eq=False, kw_only=True)
class PostprocessorDefenseConfig(PostprocessorDefenseMixin, DefensePipelineConfig):
    """Configuration for postprocessor-based defenses.

    Registers postprocessor defense behavior and plugin metadata used during
    defense runtime dispatch.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=PostprocessorDefenseMixin,
                defense_type="postprocessor",
            ),
        ],
    )


safe_store(
    group="model",
    name="postprocessor_defense",
    node=PostprocessorDefenseConfig(),
)

safe_store(
    group="search/models",
    name="postprocessor_defense",
    node=PostprocessorDefenseConfig(),
)
