"""Model defense defaults and public defense package exports.

This package provides canonical default defense config classes while preserving
backward compatibility with the existing defense runtime implementation in
model.defend.
"""

from dataclasses import dataclass, field

from ..defend import DefenseConfig, DefensePipelineConfig


@dataclass(kw_only=True)
class DefaultDefenseConfig(DefenseConfig):
    """Default neutral defense configuration.

    This config keeps defense disabled unless explicitly overridden.
    """

    defense_name: str | None = None
    defense_params: dict = field(default_factory=dict)
    init_params: dict = field(
        default_factory=lambda: {
            "library": "deckard",
            "type": "defense",
            "class": "baseline",
        },
    )


@dataclass(kw_only=True)
class DefaultSklearnDefenseConfig(DefaultDefenseConfig):
    """Default sklearn defense configuration."""

    init_params: dict = field(
        default_factory=lambda: {
            "library": "art",
            "type": "defense",
            "class": "sklearn.baseline",
        },
    )


@dataclass(kw_only=True)
class DefaultPytorchDefenseConfig(DefaultDefenseConfig):
    """Default pytorch defense configuration."""

    init_params: dict = field(
        default_factory=lambda: {
            "library": "art",
            "type": "defense",
            "class": "pytorch.baseline",
        },
    )


__all__ = [
    "DefenseConfig",
    "DefensePipelineConfig",
    "DefaultDefenseConfig",
    "DefaultSklearnDefenseConfig",
    "DefaultPytorchDefenseConfig",
]
