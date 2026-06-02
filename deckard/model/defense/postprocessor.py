"""Configuration for postprocessor defenses (output transformation)."""

from dataclasses import dataclass

from ...data import DataConfig
from ...types import ArtEsimtator, EstimatorLike, StringifiedClass
from ...utils import BaseConfig, safe_store
from .base import ARTDefenseBehaviorMixin, DefenseInitParamValue


@dataclass(eq=False, kw_only=True)
class PostprocessorDefenseConfig(ARTDefenseBehaviorMixin, BaseConfig):
    """Configuration for postprocessor-based defenses.

    Registers postprocessor defense behavior and plugin metadata used during
    defense runtime dispatch.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def postprocess(
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
        """Public verb-form entrypoint for postprocessor defense execution.

        Args:
            data: Optional runtime data config used by postprocessor defenses.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical postprocessor subtype token.
            defense_class: Resolved postprocessor defense class.
            art_class: ART wrapper class for defense execution.
            init_params: ART wrapper initialization kwargs.
            base_estimator: Base estimator to defend.
            existing_preprocessors: Existing preprocessor defenses.
            existing_postprocessors: Existing postprocessor defenses.

        Returns:
            Tuple of configured defense object and defended estimator.
        """
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
        """Attach postprocessor defense and return defense with defended estimator.

        Args:
            data: Optional runtime data config used by postprocessor defenses.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical postprocessor subtype token.
            defense_class: Resolved postprocessor defense class.
            art_class: ART wrapper class for defense execution.
            init_params: ART wrapper initialization kwargs.
            base_estimator: Base estimator to defend.
            existing_preprocessors: Existing preprocessor defenses.
            existing_postprocessors: Existing postprocessor defenses.

        Returns:
            Tuple of configured defense object and defended estimator.
        """
        _ = (data, defense_type, defense_subtype)
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
