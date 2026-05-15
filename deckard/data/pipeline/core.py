"""Core data pipeline config definitions."""

from dataclasses import dataclass, field

from ..base import DataPipelineConfig


@dataclass(eq=False, kw_only=True)
class DefaultDataPipelineConfig(DataPipelineConfig):
    """Default no-op data pipeline config."""

    pipeline: dict = field(default_factory=dict)


@dataclass(eq=False, kw_only=True)
class AnjanaDataPipelineConfig(DataPipelineConfig):
    """Pipeline config marker for anjana-family data flows."""

    pipeline: dict = field(default_factory=dict)


@dataclass(eq=False, kw_only=True)
class FairlearnDataPipelineConfig(DataPipelineConfig):
    """Pipeline config marker for fairlearn-family data flows."""

    pipeline: dict = field(default_factory=dict)


__all__ = [
    "DataPipelineConfig",
    "DefaultDataPipelineConfig",
    "AnjanaDataPipelineConfig",
    "FairlearnDataPipelineConfig",
]
