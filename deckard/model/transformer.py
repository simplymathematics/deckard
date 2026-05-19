"""Configuration for transformer defenses (feature transformation)."""

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


class _TransformerDefenseMixin(_DefenseMixin):
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
class TransformerDefenseConfig(_TransformerDefenseMixin, DefensePipelineConfig):
    """Configuration for transformer-based defenses.

    Initialization params
    ---------------------
    defense_name : str | None
        Defense class path inherited from ``DefensePipelineConfig``.
    defense_params : dict[str, Any]
        Constructor kwargs forwarded to resolved transformer defense class.
    init_params : dict[str, Any]
        Runtime ART-wrapper kwargs resolved by defense orchestration.
    plugins : list[DefenseTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``DefenseTypePlugin`` configured with:
        ``mixin_type: type = _TransformerDefenseMixin`` and
        ``defense_type: str = 'transformer'``.

    Runtime params
    --------------
    _TransformerDefenseMixin.__call__(self, *, data: Any, defense_type: str | None, defense_subtype: str | None, defense_class: Any, art_class: Any, init_params: dict, base_estimator: Any, existing_preprocessors: list, existing_postprocessors: list) -> tuple[Any, Any]
        Runtime dispatch entrypoint invoked by defense orchestration.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=_TransformerDefenseMixin,
                defense_type="transformer",
            )
        ]
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
