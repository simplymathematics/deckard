"""Model defense defaults and public defense package exports."""

from dataclasses import dataclass, field

from .base import DefenseConfig, DefensePipelineConfig


@dataclass(kw_only=True)
class DefaultDefenseConfig(DefenseConfig):
    """Default neutral defense configuration.
    
    This config keeps defense disabled unless explicitly overridden.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    defense_name: str | None = None
    defense_params: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class DefaultSklearnDefenseConfig(DefaultDefenseConfig):
    """Default sklearn defense configuration.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    defense_name: str | None = None
    defense_params: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class DefaultPytorchDefenseConfig(DefaultDefenseConfig):
    """Default pytorch defense configuration.
    
    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    defense_name: str | None = None
    defense_params: dict = field(default_factory=dict)


__all__ = [
    "DefenseConfig",
    "DefensePipelineConfig",
    "DefaultDefenseConfig",
    "DefaultSklearnDefenseConfig",
    "DefaultPytorchDefenseConfig",
]
