"""Configuration for detector defenses (adversarial detector)."""

from dataclasses import dataclass

from ...data import DataConfig
from ...types import ArtEsimtator, EstimatorLike, StringifiedClass
from ...utils import BaseConfig, safe_store
from .base import (
    ARTDefenseBehaviorMixin,
    DefenseInitParamValue,
    _is_art_torch_wrapper,
    _is_torch_model_instance,
)


@dataclass(eq=False)
class DetectorDefenseConfig(ARTDefenseBehaviorMixin, BaseConfig):
    """Configuration for detector-based defenses.

    This wraps detector-family defense behavior and registers detector-specific
    defense type plugins for runtime dispatch.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

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
        """Public verb-form entrypoint for detector defense execution.

        Args:
            data: Optional runtime data config used by detector defenses.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical detector subtype token.
            defense_class: Resolved detector defense class.
            art_class: ART wrapper class for detector execution.
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

    def detect_evasion(
        self,
        *,
        defense_class: type,
        art_class: ArtEsimtator,
        init_params: dict[str, DefenseInitParamValue],
        base_estimator: EstimatorLike,
        existing_preprocessors: list,
        existing_postprocessors: list,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Execute detector evasion defense path.

        Args:
            defense_class: Resolved detector defense class.
            art_class: ART wrapper class for detector execution.
            init_params: ART wrapper initialization kwargs.
            base_estimator: Base estimator to defend.
            existing_preprocessors: Existing preprocessor defenses.
            existing_postprocessors: Existing postprocessor defenses.

        Returns:
            Tuple of configured detector defense and defended estimator.

        Raises:
            ValueError: If model type is unsupported for evasion detectors.
        """
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

    def detect_poison(
        self,
        *,
        defense_class: type,
        init_params: dict[str, DefenseInitParamValue],
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Execute detector poison defense path.

        Args:
            defense_class: Resolved detector defense class.
            init_params: Defense execution kwargs.

        Returns:
            Tuple of configured detector defense and defended estimator.
        """
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
            data: Optional runtime data config used by detector defenses.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical detector subtype token.
            defense_class: Resolved detector defense class.
            art_class: ART wrapper class for detector execution.
            init_params: ART wrapper initialization kwargs.
            base_estimator: Base estimator to defend.
            existing_preprocessors: Existing preprocessor defenses.
            existing_postprocessors: Existing postprocessor defenses.

        Returns:
            Tuple of configured defense object and defended estimator.

        Raises:
            NotImplementedError: If detector subtype is unsupported.
        """
        _ = (data, defense_type)
        assert defense_class is not None
        subtype = (defense_subtype or "").lower()
        if subtype == "evasion":
            return self.detect_evasion(
                defense_class=defense_class,
                art_class=art_class,
                init_params=init_params,
                base_estimator=base_estimator,
                existing_preprocessors=existing_preprocessors,
                existing_postprocessors=existing_postprocessors,
            )

        if subtype == "poison":
            return self.detect_poison(
                defense_class=defense_class,
                init_params=init_params,
            )

        raise NotImplementedError(
            f"Detector subtype '{defense_subtype}' is not implemented yet.",
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
