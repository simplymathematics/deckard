"""Configuration for detector defenses (adversarial detector)."""

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


class DetectorDefenseMixin(DefenseMixin):
    """Reusable detector defense behavior."""

    def detect(
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
        """Public verb-form entrypoint for detector defense execution."""
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

    def detector_evasion(
        self,
        *,
        defense_class: type,
        art_class: ArtEsimtator,
        init_params: dict[str, DefenseInitParamValue],
        base_estimator: EstimatorLike,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Public subtype-mirroring alias for detector.evasion execution."""
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

    def detector_poison(
        self,
        *,
        defense_class: type,
        init_params: dict[str, DefenseInitParamValue],
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Public subtype-mirroring alias for detector.poison execution."""
        defense = defense_class(**(self.defense_params or {}))
        defended_estimator = defense(
            self.get_model(),
            **init_params,
        )
        return defense, defended_estimator

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
        """Build detector defense wrapper and return defense with defended estimator.

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
            Detector defense object and defended estimator.

        Raises:
            ValueError: If subtype/model combination is unsupported.
            NotImplementedError: If detector subtype runtime is not implemented.
        """
        _ = (data, defense_type)
        assert defense_class is not None
        subtype = (defense_subtype or "").lower()
        if subtype == "evasion":
            return self.detector_evasion(
                defense_class=defense_class,
                art_class=art_class,
                init_params=init_params,
                base_estimator=base_estimator,
                existing_preprocessors=existing_preprocessors,
                existing_postprocessors=existing_postprocessors,
            )

        if subtype == "poison":
            return self.detector_poison(
                defense_class=defense_class,
                init_params=init_params,
            )

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
