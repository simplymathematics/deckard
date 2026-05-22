"""Backward-compatible re-exports for legacy pipeline.core targets.

Prefer importing from ``deckard.data.pipeline.base``.
"""

from .base import (  # noqa: F401
    AnjanaDataPipelineConfig,
    DataPipeline,
    DataPipelineConfig,
    DefaultDataPipelineConfig,
    FairlearnDataPipelineConfig,
)

__all__ = [
    "DataPipeline",
    "DataPipelineConfig",
    "DefaultDataPipelineConfig",
    "AnjanaDataPipelineConfig",
    "FairlearnDataPipelineConfig",
]
