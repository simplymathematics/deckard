"""Configuration for trainer defenses (adversarial training)."""

from dataclasses import dataclass, field
from typing import Any

from deckard.plugins.defense import DefenseTypePlugin
from .defend import (
    DefensePipelineConfig,
    _DefenseMixin,
    _is_art_torch_wrapper,
    _is_torch_model_instance,
)
from ..utils import safe_store


class _TrainerDefenseMixin(_DefenseMixin):
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
class TrainerDefenseConfig(_TrainerDefenseMixin, DefensePipelineConfig):
    """Configuration for trainer-based defenses (adversarial training).

    Initialization params
    ---------------------
    defense_name : str | None
        Defense class path inherited from ``DefensePipelineConfig``.
    defense_params : dict[str, Any]
        Constructor kwargs forwarded to resolved trainer defense class.
    init_params : dict[str, Any]
        Runtime ART-wrapper kwargs resolved by defense orchestration.
    plugins : list[DefenseTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``DefenseTypePlugin`` configured with:
        ``mixin_type: type = _TrainerDefenseMixin`` and
        ``defense_type: str = 'trainer'``.

    Runtime params
    --------------
    _TrainerDefenseMixin.__call__(self, *, data: Any, defense_type: str | None, defense_subtype: str | None, defense_class: Any, art_class: Any, init_params: dict, base_estimator: Any, existing_preprocessors: list, existing_postprocessors: list) -> tuple[Any, Any]
        Runtime dispatch entrypoint invoked by defense orchestration.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=_TrainerDefenseMixin,
                defense_type="trainer",
            ),
        ],
    )


# Register trainer defense config
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
