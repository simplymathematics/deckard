"""Configuration for trainer defenses (adversarial training)."""

from dataclasses import dataclass

from ...data import DataConfig
from ...types import ArtEsimtator, EstimatorLike, StringifiedClass
from ...utils import BaseConfig, safe_store
from .base import (
    ARTDefenseBehaviorMixin,
    DefenseInitParamValue,
    _dispatch_runtime_callable,
    _is_art_torch_wrapper,
    _is_torch_model_instance,
)


@dataclass(eq=False, kw_only=True)
class TrainerDefenseConfig(ARTDefenseBehaviorMixin, BaseConfig):
    """Configuration for trainer-based defenses.

    Registers trainer defense behavior and plugin metadata used during defense
    runtime dispatch.

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
            data: Optional runtime data config used by trainer defenses.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical trainer subtype token.
            defense_class: Resolved trainer defense class.
            art_class: ART wrapper class for defense execution.
            init_params: ART wrapper initialization kwargs.
            base_estimator: Base estimator to defend.
            existing_preprocessors: Existing preprocessor defenses.
            existing_postprocessors: Existing postprocessor defenses.

        Returns:
            Tuple of configured defense object and defended estimator.
        """
        return _dispatch_runtime_callable(self, **locals())

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
            data: Optional runtime data config used by trainer defenses.
            defense_type: Canonical defense family token.
            defense_subtype: Canonical trainer subtype token.
            defense_class: Resolved trainer defense class.
            art_class: ART wrapper class for defense execution.
            init_params: ART wrapper initialization kwargs.
            base_estimator: Base estimator to defend.
            existing_preprocessors: Existing preprocessor defenses.
            existing_postprocessors: Existing postprocessor defenses.

        Returns:
            Tuple of configured defense object and defended estimator.

        Raises:
            ValueError: If base estimator type is unsupported for trainer defenses.
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
