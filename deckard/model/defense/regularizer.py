"""Configuration for regularizer defenses (training-time regularization)."""

from dataclasses import dataclass

from ...data import DataConfig
from ...frameworks.types import ArtEsimtator, EstimatorLike, StringifiedClass
from ...utils import BaseConfig, safe_store
from .base import ARTDefenseBehaviorMixin, DefenseInitParamValue


@dataclass(eq=False, kw_only=True)
class RegularizerDefenseConfig(ARTDefenseBehaviorMixin, BaseConfig):
    """Configuration for regularizer-based defenses.

    Registers regularizer defense behavior and plugin metadata used during
    defense runtime dispatch.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def regularize(
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
        """Public ergonomic alias for regularizer defense execution.

        Args:
            data: Optional runtime data config used by regularizer defenses.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical regularizer subtype token.
            defense_class: Resolved regularizer defense class.
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
        """Raise for unimplemented regularizer defense runtime path.

        Args:
            data: Optional runtime data config used by regularizer defenses.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical regularizer subtype token.
            defense_class: Resolved regularizer defense class.
            art_class: ART wrapper class for defense execution.
            init_params: ART wrapper initialization kwargs.
            base_estimator: Base estimator to defend.
            existing_preprocessors: Existing preprocessor defenses.
            existing_postprocessors: Existing postprocessor defenses.

        Returns:
            Tuple of configured defense object and defended estimator.

        Raises:
            NotImplementedError: Regularizer runtime defense behavior is not implemented.
        """
        _ = (
            data,
            defense_type,
            defense_subtype,
            defense_class,
            art_class,
            init_params,
            base_estimator,
            existing_preprocessors,
            existing_postprocessors,
        )
        raise NotImplementedError(
            "Regularizer defenses are not implemented yet.",
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
