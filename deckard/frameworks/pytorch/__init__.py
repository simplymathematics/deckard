"""Pytorch framework package for framework-specific config implementations."""

from typing import TYPE_CHECKING

from .defense import DefaultPytorchDefenseConfig

if TYPE_CHECKING:  # pragma: no cover
    from .attack import PytorchAttackConfig
    from .data import (
        PytorchCustomDataConfig,
        PytorchDataConfig,
        PytorchDataPipelineConfig,
    )
    from .experiment import TorchExperimentConfig
    from .fairness_data import FairlearnPytorchDataConfig
    from .model import PytorchModelConfig

__all__ = [
    "PytorchDataConfig",
    "PytorchCustomDataConfig",
    "PytorchDataPipelineConfig",
    "FairlearnPytorchDataConfig",
    "PytorchModelConfig",
    "PytorchAttackConfig",
    "DefaultPytorchDefenseConfig",
    "TorchExperimentConfig",
]


def __getattr__(name):
    if name in {
        "PytorchDataConfig",
        "PytorchCustomDataConfig",
        "PytorchDataPipelineConfig",
    }:
        from .data import (
            PytorchCustomDataConfig,
            PytorchDataConfig,
            PytorchDataPipelineConfig,
        )

        mapping = {
            "PytorchDataConfig": PytorchDataConfig,
            "PytorchCustomDataConfig": PytorchCustomDataConfig,
            "PytorchDataPipelineConfig": PytorchDataPipelineConfig,
        }
        return mapping[name]
    if name == "FairlearnPytorchDataConfig":
        from .fairness_data import FairlearnPytorchDataConfig

        return FairlearnPytorchDataConfig
    if name == "PytorchModelConfig":
        from .model import PytorchModelConfig

        return PytorchModelConfig
    if name == "PytorchAttackConfig":
        from .attack import PytorchAttackConfig

        return PytorchAttackConfig
    if name == "TorchExperimentConfig":
        from .experiment import TorchExperimentConfig

        return TorchExperimentConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
