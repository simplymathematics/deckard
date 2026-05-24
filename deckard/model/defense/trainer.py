"""Configuration for trainer defenses (adversarial training)."""

from dataclasses import dataclass, field
from typing import Any

from deckard.plugins.defense import DefenseTypePlugin

from ...utils import safe_store
from .base import (
    DefensePipelineConfig,
    DefenseMixin,
    _is_art_torch_wrapper,
    _is_torch_model_instance,
)


class TrainerDefenseMixin(DefenseMixin):
    """Reusable trainer defense behavior (adversarial training)."""

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
    group="model",
    name="trainer_defense",
    node=TrainerDefenseConfig(),
)

safe_store(
    group="search/models",
    name="trainer_defense",
    node=TrainerDefenseConfig(),
)
