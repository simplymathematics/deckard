"""Configuration for transformer defenses (feature transformation)."""

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


class TransformerDefenseMixin(DefenseMixin):
    """Reusable transformer defense behavior.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

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
        """Build transformer defense wrapper and return defended estimator.

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
            Instantiated transformer defense object and defended estimator.

        Raises:
            ValueError: If subtype is unknown or estimator type is unsupported.
        """
        _ = (data, defense_type)
        assert defense_class is not None
        transformer_params = dict(self.defense_params or {})
        subtype = (defense_subtype or "").lower()
        if subtype not in {"evasion", "poisoning"}:
            raise ValueError(f"Unknown transformer subtype: {defense_subtype}")

        if not _is_torch_model_instance(base_estimator) and not _is_art_torch_wrapper(
            self._model,
        ):
            raise ValueError(
                "Transformer defenses only support neural-network models. "
                f"Got base estimator type {type(base_estimator)}.",
            )

        transformer_classifier = self._build_art_wrapper(
            art_class=art_class,
            base_estimator=base_estimator,
            init_params=init_params,
            preprocessing_defences=existing_preprocessors,
            postprocessing_defences=existing_postprocessors,
        )

        try:
            defense = defense_class(
                classifier=transformer_classifier,
                **transformer_params,
            )
        except TypeError:
            try:
                defense = defense_class(
                    transformer_classifier,
                    **transformer_params,
                )
            except NotImplementedError as exc:
                raise ValueError(
                    "Transformer defense initialization failed for the current "
                    "ART classifier backend. Ensure the estimator type is "
                    "supported by the selected defense.",
                ) from exc
        except NotImplementedError as exc:
            raise ValueError(
                "Transformer defense initialization failed for the current "
                "ART classifier backend. Ensure the estimator type is "
                "supported by the selected defense.",
            ) from exc

        if hasattr(defense, "get_classifier"):
            defended_estimator = defense.get_classifier()
        else:
            defended_estimator = transformer_classifier
        return defense, defended_estimator


@dataclass(eq=False, kw_only=True)
class TransformerDefenseConfig(TransformerDefenseMixin, DefensePipelineConfig):
    """Configuration for transformer-based defenses.
    
    Registers transformer defense behavior and plugin metadata used during
    defense runtime dispatch.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=TransformerDefenseMixin,
                defense_type="transformer",
            ),
        ],
    )


safe_store(
    group="model",
    name="transformer_defense",
    node=TransformerDefenseConfig(),
)

safe_store(
    group="search/models",
    name="transformer_defense",
    node=TransformerDefenseConfig(),
)
