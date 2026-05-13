"""Configuration for preprocessor defenses (input transformation)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .defend import DefensePipelineConfig, _DefenseMixin
from ..utils import safe_store

if TYPE_CHECKING:
    pass


class _PreprocessorDefenseMixin(_DefenseMixin):
    """Reusable preprocessor defense behavior."""

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
            preprocessing_defences=existing_preprocessors + [defense],
            postprocessing_defences=existing_postprocessors,
        )
        return defense, defended_estimator


@dataclass(eq=False)
class PreprocessorDefenseConfig(_PreprocessorDefenseMixin, DefensePipelineConfig):
    """
    Configuration for preprocessor-based defenses.
    
    Preprocessors apply transformations to inputs before they reach the model
    (e.g., compression, normalization, denoising). Improves robustness by
    making adversarial perturbations less effective.
    """

    pass


# Register preprocessor defense config
safe_store(
    group="model",
    name="preprocessor_defense",
    node=PreprocessorDefenseConfig(),
)

safe_store(
    group="search/models",
    name="preprocessor_defense",
    node=PreprocessorDefenseConfig(),
)
