"""Configuration for transformer defenses (feature transformation)."""

from dataclasses import dataclass, field
from typing import Any

from deckard.plugins.defense import DefenseTypePlugin

from ..utils import safe_store
from .defense.base import (
    DefensePipelineConfig,
    DefenseMixin,
    _is_art_torch_wrapper,
    _is_torch_model_instance,
)


class TransformerDefenseMixin(DefenseMixin):
    """Reusable transformer defense behavior."""

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
        """Build transformer defense wrapper and return defense with defended estimator."""
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
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=TransformerDefenseMixin,
                defense_type="transformer",
            ),
        ],
    )


# Register transformer defense config
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
