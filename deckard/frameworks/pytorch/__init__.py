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
    from .sample import (
        PytorchBaseSampler,
        PytorchFoldSampler,
        PytorchShuffleSampler,
        PytorchSplitSampler,
        TorchBaseSampler,
        TorchKFoldSampler,
        TorchShuffleSampler,
        TorchSplitSampler,
    )

__all__ = [
    "PytorchDataConfig",
    "PytorchCustomDataConfig",
    "PytorchDataPipelineConfig",
    "FairlearnPytorchDataConfig",
    "PytorchModelConfig",
    "PytorchAttackConfig",
    "DefaultPytorchDefenseConfig",
    "TorchExperimentConfig",
    "PytorchBaseSampler",
    "PytorchSplitSampler",
    "PytorchFoldSampler",
    "PytorchShuffleSampler",
    "TorchBaseSampler",
    "TorchSplitSampler",
    "TorchKFoldSampler",
    "TorchShuffleSampler",
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
    if name in {
        "PytorchBaseSampler",
        "PytorchSplitSampler",
        "PytorchFoldSampler",
        "PytorchShuffleSampler",
        "TorchBaseSampler",
        "TorchSplitSampler",
        "TorchKFoldSampler",
        "TorchShuffleSampler",
    }:
        from .sample import (
            PytorchBaseSampler,
            PytorchFoldSampler,
            PytorchShuffleSampler,
            PytorchSplitSampler,
            TorchBaseSampler,
            TorchKFoldSampler,
            TorchShuffleSampler,
            TorchSplitSampler,
        )

        mapping = {
            "PytorchBaseSampler": PytorchBaseSampler,
            "PytorchSplitSampler": PytorchSplitSampler,
            "PytorchFoldSampler": PytorchFoldSampler,
            "PytorchShuffleSampler": PytorchShuffleSampler,
            "TorchBaseSampler": TorchBaseSampler,
            "TorchSplitSampler": TorchSplitSampler,
            "TorchKFoldSampler": TorchKFoldSampler,
            "TorchShuffleSampler": TorchShuffleSampler,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
