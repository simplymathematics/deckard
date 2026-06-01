from dataclasses import dataclass, field
from typing import Any, Callable, Union
from ..frameworks.types import StringifiedClass
from ..frameworks.types import EstimatorLike
from ..utils import BaseConfig

DefenseScalar = str | int | float | bool | None
DefenseValue = DefenseScalar | list["DefenseValue"] | dict[str, "DefenseValue"]
DefenseHandler = Callable[..., tuple[BaseConfig | None, EstimatorLike]]


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

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    mixin_type: Any
    defense_type: StringifiedClass | None
    defense_subtype: Union[str, None] = None
    excluded_subtypes: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"help": "Configuration field: excluded_subtypes."},
    )
    init_params: dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Configuration field: init_params."}
    )

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
        defense_type: StringifiedClass | None,
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
        runtime: BaseConfig,
        *,
        defense_type: StringifiedClass | None,
        defense_subtype: Union[str, None],
        default_mixins: tuple[type, ...],
    ) -> tuple[type, ...]:
        """Return matching defense mixins for the runtime defense family/subtype.

        Args:
            runtime: Runtime config instance (unused in current matching logic).
            defense_type: Requested defense family.
            defense_subtype: Optional requested defense subtype.
            default_mixins: Framework default mixin set.

        Returns:
            Single-item tuple containing the configured mixin when matched, else empty tuple.
        """
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
        runtime: BaseConfig,
        *,
        defense_type: StringifiedClass | None,
        defense_subtype: Union[str, None],
        default_handler: DefenseHandler | None,
        default_mixins: tuple[type, ...],
    ) -> DefenseHandler | None:
        """Return callable defense handler when plugin matches runtime defense context.

        Args:
            runtime: Runtime config instance bound to the resolved mixin.
            defense_type: Requested defense family.
            defense_subtype: Optional requested defense subtype.
            default_handler: Framework default defense handler.
            default_mixins: Framework default defense mixins.

        Returns:
            Callable plugin handler when matched, otherwise None.
        """
        _ = (default_handler, default_mixins)
        if not self._matches(
            defense_type=defense_type,
            defense_subtype=defense_subtype,
        ):
            return None
        return lambda *args, **kwargs: self(runtime, *args, **kwargs)

    def __call__(
        self,
        runtime: BaseConfig,
        *args: DefenseValue,
        **kwargs: DefenseValue,
    ) -> tuple[BaseConfig | None, EstimatorLike]:
        """Delegate defense execution to the configured mixin bound to runtime.

        Args:
            runtime: Runtime config instance passed into the defense mixin.
            *args: Positional arguments forwarded to the defense handler.
            **kwargs: Keyword arguments forwarded to the defense handler.

        Returns:
            Two-item tuple returned by the defense handler.
        """
        mixin = self._resolve_mixin_type()
        if isinstance(mixin, type) and mixin in type(runtime).mro():
            handler = runtime
        else:
            handler = mixin(runtime)
        return handler(*args, **kwargs)
