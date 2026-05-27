"""Configuration for trainer defenses (adversarial training)."""

from dataclasses import dataclass, field

from deckard.plugins.defense import DefenseTypePlugin

from ...data import DataConfig
from ...frameworks.types import ArtEsimtator, EstimatorLike, StringifiedClass
from ...utils import BaseConfig, safe_store
from .base import (
    DefenseInitParamValue,
    DefensePipelineConfig,
    DefenseMixin,
    _is_art_torch_wrapper,
    _is_torch_model_instance,
)


class TrainerDefenseMixin(DefenseMixin):
    """Reusable trainer defense behavior (adversarial training).

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def train_defense(
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
        """Public verb-form alias for trainer defense execution.

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
            Instantiated trainer defense object and defended estimator.
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
        """Build trainer defense wrapper and return defended estimator.

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
            Instantiated trainer defense object and defended estimator.

        Raises:
            ValueError: If estimator type is unsupported for trainer defenses.
        """
        _ = (data, defense_type, defense_subtype)
        assert defense_class is not None
        trainer_params = dict(self.defense_params or {})

        if not _is_torch_model_instance(base_estimator) and not _is_art_torch_wrapper(
            self._model,
        ):
            raise ValueError(
                "Retraining trainer defenses only support neural-network models. "
                f"Got base estimator type {type(base_estimator)}.",
            )

        trainer_classifier = self._build_art_wrapper(
            art_class=art_class,
            base_estimator=base_estimator,
            init_params=init_params,
            preprocessing_defences=existing_preprocessors,
            postprocessing_defences=existing_postprocessors,
        )
        try:
            defense = defense_class(
                classifier=trainer_classifier,
                **trainer_params,
            )
        except TypeError:
            defense = defense_class(trainer_classifier, **trainer_params)

        if hasattr(defense, "get_classifier"):
            defended_estimator = defense.get_classifier()
        else:
            defended_estimator = trainer_classifier
        return defense, defended_estimator


@dataclass(eq=False, kw_only=True)
class TrainerDefenseConfig(TrainerDefenseMixin, DefensePipelineConfig):
    """Configuration for trainer-based defenses.

    Registers trainer defense behavior and plugin metadata used during defense
    runtime dispatch.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=TrainerDefenseMixin,
                defense_type="trainer",
            ),
        ],
    )


safe_store(
    group="model/defense",
    name="trainer_defense",
    node=TrainerDefenseConfig(),
)

safe_store(
    group="search/defenses",
    name="trainer_defense",
    node=TrainerDefenseConfig(),
)
