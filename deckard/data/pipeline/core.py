"""Backward-compatible re-exports for legacy pipeline.core targets.

Prefer importing from ``deckard.data.pipeline.base``.
"""

from .base import (  # noqa: F401
    DataConfig,
    DataPipeline,
)

__all__ = [
    "DataPipeline",
    "DataConfig",
]
