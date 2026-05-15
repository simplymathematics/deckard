"""Configuration for detector defenses (adversarial detector)."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deckard.plugins.defense import DefenseTypePlugin
from .defend import (
    DefensePipelineConfig,
    _DefenseMixin,
    _is_art_torch_wrapper,
    _is_torch_model_instance,
)
from ..utils import safe_store




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
    ) -> tuple[Any, Any]:
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
    """Configuration for detector-based defenses.

    Initialization params
    ---------------------
    defense_name : str | None
        Defense class path inherited from ``DefensePipelineConfig``.
    defense_params : dict[str, Any]
        Constructor kwargs forwarded to resolved detector defense class.
    init_params : dict[str, Any]
        Runtime ART-wrapper kwargs resolved by defense orchestration.
    plugins : list[DefenseTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``DefenseTypePlugin`` configured with:
        ``mixin_type: type = _DetectorDefenseMixin`` and
        ``defense_type: str = 'detector'``.

    Runtime params
    --------------
    _DetectorDefenseMixin.__call__(self, *, data: Any, defense_type: str | None, defense_subtype: str | None, defense_class: Any, art_class: Any, init_params: dict, base_estimator: Any, existing_preprocessors: list, existing_postprocessors: list) -> tuple[Any, Any]
        Runtime dispatch entrypoint invoked by ``DefenseConfig``/
        ``DefensePipelineConfig`` defense orchestration.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=_DetectorDefenseMixin,
                defense_type="detector",
            )
        ]
    )


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
