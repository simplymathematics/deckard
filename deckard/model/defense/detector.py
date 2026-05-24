"""Configuration for detector defenses (adversarial detector)."""

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


class DetectorDefenseMixin(DefenseMixin):
    """Reusable detector defense behavior."""

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
        subtype = (defense_subtype or "").lower()
        if subtype == "evasion":
            if not _is_torch_model_instance(
                base_estimator,
            ) and not _is_art_torch_wrapper(
                self._model,
            ):
                raise ValueError(
                    "Evasion detector defenses only support neural-network models. "
                    f"Got base estimator type {type(base_estimator)}.",
                )

            detector_classifier = self._build_art_wrapper(
                art_class=art_class,
                base_estimator=base_estimator,
                init_params=init_params,
                preprocessing_defences=existing_preprocessors,
                postprocessing_defences=existing_postprocessors,
            )

            detector_params = dict(self.defense_params or {})
            try:
                defense = defense_class(
                    detector=detector_classifier,
                    **detector_params,
                )
            except TypeError:
                defense = defense_class(
                    detector_classifier,
                    **detector_params,
                )

            setattr(detector_classifier, "_deckard_evasion_detector", defense)
            return defense, detector_classifier

        if subtype == "poison":
            defense = defense_class(**(self.defense_params or {}))
            defended_estimator = defense(
                self.get_model(),
                **init_params,
            )
            return defense, defended_estimator

        raise NotImplementedError(
            f"Detector subtype '{defense_subtype}' is not implemented yet.",
        )


@dataclass(eq=False)
class DetectorDefenseConfig(DetectorDefenseMixin, DefensePipelineConfig):
    """Configuration for detector-based defenses.

    This wraps detector-family defense behavior and registers detector-specific
    defense type plugins for runtime dispatch.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=DetectorDefenseMixin,
                defense_type="detector",
            ),
        ],
    )


safe_store(
    group="model",
    name="detector_defense",
    node=DetectorDefenseConfig(),
)

safe_store(
    group="search/models",
    name="detector_defense",
    node=DetectorDefenseConfig(),
)
