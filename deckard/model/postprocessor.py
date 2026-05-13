"""Configuration for postprocessor defenses (output transformation)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .defend import DefensePipelineConfig, _DefenseMixin
from ..utils import safe_store

if TYPE_CHECKING:
    pass


class _PostprocessorDefenseMixin(_DefenseMixin):
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
    ):
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


@dataclass(eq=False)
class PostprocessorDefenseConfig(_PostprocessorDefenseMixin, DefensePipelineConfig):
    """
    Configuration for postprocessor-based defenses.
    
    Postprocessors apply transformations to model outputs before returning them
    (e.g., confidence thresholding, output manipulation). Defends against
    attacks by modifying the final predictions.
    """

    pass


# Register postprocessor defense config
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
