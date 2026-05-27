"""Configuration for regularizer defenses (training-time regularization)."""

from dataclasses import dataclass, field

from deckard.plugins.defense import DefenseTypePlugin

from ...data import DataConfig
from ...frameworks.types import ArtEsimtator, EstimatorLike, StringifiedClass
from ...utils import BaseConfig, safe_store
from .base import DefenseInitParamValue, DefensePipelineConfig, DefenseMixin


class RegularizerDefenseMixin(DefenseMixin):
    """Reusable regularizer defense behavior.

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
        """Public ergonomic alias for regularizer defense execution."""
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
            Placeholder regularizer defense payload and estimator.

        Raises:
            NotImplementedError: Always raised until regularizer defense runtime is implemented.
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


@dataclass(eq=False, kw_only=True)
class RegularizerDefenseConfig(RegularizerDefenseMixin, DefensePipelineConfig):
    """Configuration for regularizer-based defenses.

    Registers regularizer defense behavior and plugin metadata used during
    defense runtime dispatch.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
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
