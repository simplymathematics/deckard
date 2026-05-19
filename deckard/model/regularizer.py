"""Configuration for regularizer defenses (training-time regularization)."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deckard.plugins.defense import DefenseTypePlugin
from .defend import DefensePipelineConfig, _DefenseMixin
from ..utils import safe_store


class _RegularizerDefenseMixin(_DefenseMixin):
    """Reusable regularizer defense behavior."""

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
        raise NotImplementedError(
            "Regularizer defenses are not implemented yet.",
        )


@dataclass(eq=False, kw_only=True)
class RegularizerDefenseConfig(_RegularizerDefenseMixin, DefensePipelineConfig):
    """Configuration for regularizer-based defenses.

    Initialization params
    ---------------------
    defense_name : str | None
        Defense class path inherited from ``DefensePipelineConfig``.
    defense_params : dict[str, Any]
        Constructor kwargs forwarded to resolved regularizer defense class.
    init_params : dict[str, Any]
        Runtime ART-wrapper kwargs resolved by defense orchestration.
    plugins : list[DefenseTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``DefenseTypePlugin`` configured with:
        ``mixin_type: type = _RegularizerDefenseMixin`` and
        ``defense_type: str = 'regularizer'``.

    Runtime params
    --------------
    _RegularizerDefenseMixin.__call__(self, *, data: Any, defense_type: str | None, defense_subtype: str | None, defense_class: Any, art_class: Any, init_params: dict, base_estimator: Any, existing_preprocessors: list, existing_postprocessors: list) -> tuple[Any, Any]
        Runtime dispatch entrypoint invoked by defense orchestration.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=_RegularizerDefenseMixin,
                defense_type="regularizer",
            )
        ]
    )


# Register regularizer defense config
safe_store(
    group="model",
    name="regularizer_defense",
    node=RegularizerDefenseConfig(),
)

safe_store(
    group="search/models",
    name="regularizer_defense",
    node=RegularizerDefenseConfig(),
)
