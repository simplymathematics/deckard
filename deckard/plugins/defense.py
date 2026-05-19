from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(eq=False, kw_only=True)
class DefenseTypePlugin:
    """Generic defense plugin that binds one mixin to one defense family/subtype.

    Initialization fields
    ---------------------
    mixin_type : Any
        Mixin class (or import path) implementing runtime ``__call__``.
    defense_type : str | None
        Defense family this plugin matches.
    defense_subtype : str | None
        Optional subtype constraint.
    excluded_subtypes : tuple[str, ...]
        Subtypes explicitly excluded from this plugin match.
    init_params : dict[str, Any]
        Metadata-only declaration payload for class/type/library docs.
    """

    mixin_type: Any
    defense_type: Union[str, None]
    defense_subtype: Union[str, None] = None
    excluded_subtypes: tuple[str, ...] = field(default_factory=tuple)
    init_params: dict[str, Any] = field(default_factory=dict)

    def _resolve_mixin_type(self) -> type:
        if isinstance(self.mixin_type, str):
            # Assume resolve_class is available in the runtime context
            from deckard.utils import resolve_class

            resolved = resolve_class(self.mixin_type)
            self.mixin_type = resolved
            return resolved
        return self.mixin_type

    def _matches(
        self,
        *,
        defense_type: Union[str, None],
        defense_subtype: Union[str, None],
    ) -> bool:
        if (defense_type or "").lower() != (self.defense_type or "").lower():
            return False
        subtype = (defense_subtype or "").lower()
        if (
            self.defense_subtype is not None
            and subtype != self.defense_subtype.lower()
        ):
            return False
        if subtype in {item.lower() for item in self.excluded_subtypes}:
            return False
        return True

    def resolve_defense_mixins(
        self,
        runtime: Any,
        *,
        defense_type: Union[str, None],
        defense_subtype: Union[str, None],
        default_mixins: tuple[type, ...],
    ) -> tuple[type, ...]:
        _ = (runtime, default_mixins)
        if not self._matches(
            defense_type=defense_type,
            defense_subtype=defense_subtype,
        ):
            return ()
        mixin = self._resolve_mixin_type()
        return (mixin,)

    def resolve_defense_handler(
        self,
        runtime: Any,
        *,
        defense_type: Union[str, None],
        defense_subtype: Union[str, None],
        default_handler: Any,
        default_mixins: tuple[type, ...],
    ) -> Any:
        _ = (default_handler, default_mixins)
        if not self._matches(
            defense_type=defense_type,
            defense_subtype=defense_subtype,
        ):
            return None
        return lambda *args, **kwargs: self(runtime, *args, **kwargs)

    def __call__(self, runtime: Any, *args, **kwargs) -> tuple[Any, Any]:
        mixin = self._resolve_mixin_type()
        handler = mixin(runtime)
        return handler(*args, **kwargs)
