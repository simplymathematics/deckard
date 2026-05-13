"""Configuration for detector defenses (adversarial detector)."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .defend import (
    DefensePipelineConfig,
    _DefenseMixin,
    _is_art_torch_wrapper,
    _is_torch_model_instance,
)
from ..utils import safe_store

if TYPE_CHECKING:
    pass


class _DetectorDefenseMixin(_DefenseMixin):
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
    ):
        assert defense_class is not None
        subtype = (defense_subtype or "").lower()
        if subtype == "evasion":
            if not _is_torch_model_instance(base_estimator) and not _is_art_torch_wrapper(
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
class DetectorDefenseConfig(_DetectorDefenseMixin, DefensePipelineConfig):
    """
    Configuration for detector-based defenses.
    
    Detectors identify and reject adversarial examples at test time
    without modifying the model itself. Uses auxiliary detection models
    to flag suspicious inputs. Generally optimized against evasion or poisoning attacks, 
    but your mileage may vary with other attacks.
    """

    pass


# Register detector defense config
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
