"""Abstract framework contracts for Deckard framework-specific configs.

These ABCs define the minimum public API expected from framework-specific
config implementations while preserving the _Mixin -> _Plugin -> Config pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FrameworkDataConfig(ABC):
    """Framework-level data config contract."""

    @abstractmethod
    def load_data(self) -> tuple[Any, Any]:
        """Return feature and target objects for the configured dataset."""

    @abstractmethod
    def sample_data(self, X: Any, y: Any) -> tuple[Any, Any, Any, Any]:
        """Return train/test (and optional validation) splits."""


class FrameworkModelConfig(ABC):
    """Framework-level model config contract."""

    @abstractmethod
    def build_model(self, data: Any) -> Any:
        """Construct a model instance using data-derived context."""

    @abstractmethod
    def fit_model(self, data: Any) -> Any:
        """Train and return the model runtime object."""


class FrameworkAttackConfig(ABC):
    """Framework-level attack config contract."""

    @abstractmethod
    def build_attack(self, model: Any, data: Any) -> Any:
        """Construct attack runtime using model and data context."""


class FrameworkDetectorConfig(ABC):
    """Framework-level detector config contract."""

    @abstractmethod
    def build_detector(self, model: Any, attack: Any) -> Any:
        """Construct detector runtime for model/attack outputs."""


class FrameworkExperimentConfig(ABC):
    """Framework-level experiment orchestration contract."""

    @abstractmethod
    def run_experiment(self) -> dict[str, Any]:
        """Execute experiment and return score/metric dictionary."""


class FrameworkScorerConfig(ABC):
    """Framework-level scorer contract."""

    @abstractmethod
    def score(self, *, data: Any = None, model: Any = None, attack: Any = None) -> dict:
        """Compute and return metric dictionary from runtime context."""