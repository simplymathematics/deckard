"""Configuration for preprocessor defenses (input transformation)."""

from dataclasses import dataclass, field
from typing import Any

from deckard.plugins.defense import DefenseTypePlugin

from ..utils import safe_store
from .defend import DefensePipelineConfig, _DefenseMixin


class _PreprocessorDefenseMixin(_DefenseMixin):
    """Reusable preprocessor defense behavior."""

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
        defense = defense_class(**(self.defense_params or {}))
        defended_estimator = self._build_art_wrapper(
            art_class=art_class,
            base_estimator=base_estimator,
            init_params=init_params,
            preprocessing_defences=existing_preprocessors + [defense],
            postprocessing_defences=existing_postprocessors,
        )
        return defense, defended_estimator


@dataclass(eq=False, kw_only=True)
class PreprocessorDefenseConfig(_PreprocessorDefenseMixin, DefensePipelineConfig):
    """Configuration for preprocessor-based defenses.

    Initialization params
    ---------------------
    defense_name : str | None
        Defense class path inherited from ``DefensePipelineConfig``.
    defense_params : dict[str, Any]
        Constructor kwargs forwarded to resolved preprocessor defense class.
    init_params : dict[str, Any]
        Runtime ART-wrapper kwargs resolved by defense orchestration.
    plugins : list[DefenseTypePlugin]
        Declarative runtime plugin specs. Default contains one
        ``DefenseTypePlugin`` configured with:
        ``mixin_type: type = _PreprocessorDefenseMixin`` and
        ``defense_type: str = 'preprocessor'``.

    Runtime params
    --------------
    _PreprocessorDefenseMixin.__call__(self, *, data: Any, defense_type: str | None, defense_subtype: str | None, defense_class: Any, art_class: Any, init_params: dict, base_estimator: Any, existing_preprocessors: list, existing_postprocessors: list) -> tuple[Any, Any]
        Runtime dispatch entrypoint invoked by defense orchestration.
    """

    plugins: list = field(
        default_factory=lambda: [
            DefenseTypePlugin(
                mixin_type=_PreprocessorDefenseMixin,
                defense_type="preprocessor",
            ),
        ],
    )


# Register preprocessor defense config
safe_store(
    group="model",
    name="preprocessor_defense",
    node=PreprocessorDefenseConfig(),
)

safe_store(
    group="search/models",
    name="preprocessor_defense",
    node=PreprocessorDefenseConfig(),
)
