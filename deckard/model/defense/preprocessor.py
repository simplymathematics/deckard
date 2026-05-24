"""Configuration for preprocessor defenses (input transformation)."""

from dataclasses import dataclass, field

from deckard.plugins.defense import DefenseTypePlugin

from ...data import DataConfig
from ...frameworks.types import ArtEsimtator, EstimatorLike, StringifiedClass
from ...utils import BaseConfig, safe_store
from .base import DefenseInitParamValue, DefensePipelineConfig, DefenseMixin


class PreprocessorDefenseMixin(DefenseMixin):
    """Reusable preprocessor defense behavior."""

    def preprocess(
        self,
        *,
        data: DataConfig | None,
        defense_type: StringifiedClass | None,
        defense_subtype: str | None,
        defense_class: type | None,
        art_class: ArtEsimtator,
        init_params: dict[str, DefenseInitParamValue],
        base_estimator: EstimatorLike,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Public verb-form entrypoint for preprocessor defense execution."""
        return self(
            data=data,
            defense_type=defense_type,
            defense_subtype=defense_subtype,
            defense_class=defense_class,
            art_class=art_class,
            init_params=init_params,
            base_estimator=base_estimator,
            existing_preprocessors=existing_preprocessors,
            existing_postprocessors=existing_postprocessors,
        )

    def __call__(
        self,
        *,
        data: DataConfig | None,
        defense_type: StringifiedClass | None,
        defense_subtype: str | None,
        defense_class: type | None,
        art_class: ArtEsimtator,
        init_params: dict[str, DefenseInitParamValue],
        base_estimator: EstimatorLike,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Attach preprocessor defense and return defense with defended estimator.

        Args:
            data: Data runtime payload.
            defense_type: Parsed defense family token.
            defense_subtype: Parsed defense subtype token.
            defense_class: Concrete defense class resolved from defense name.
            art_class: ART estimator wrapper class selected for model type.
            init_params: Runtime ART estimator initialization kwargs.
            base_estimator: Unwrapped model estimator used as defense target.
            existing_preprocessors: Existing preprocessor defenses already attached.
            existing_postprocessors: Existing postprocessor defenses already attached.

        Returns:
            Preprocessor defense object and defended estimator.
        """
        _ = (data, defense_type, defense_subtype)
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


@dataclass(eq=False, kw_only=True)
class PreprocessorDefenseConfig(PreprocessorDefenseMixin, DefensePipelineConfig):
    """Configuration for preprocessor-based defenses.

    Registers preprocessor defense behavior and plugin metadata used during
    defense runtime dispatch.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=PreprocessorDefenseMixin,
                defense_type="preprocessor",
            ),
        ],
    )


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
