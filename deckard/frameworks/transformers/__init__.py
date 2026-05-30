"""Transformers framework package for Hugging Face model wrappers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .declarations import GenericFlexibleTransformer, HuggingFaceArtModelWrapper
    from .model import HuggingFacePytorchModelConfig

__all__ = [
    "GenericFlexibleTransformer",
    "HuggingFaceArtModelWrapper",
    "HuggingFacePytorchModelConfig",
]


def __getattr__(name):
    if name == "GenericFlexibleTransformer":
        from .declarations import GenericFlexibleTransformer

        return GenericFlexibleTransformer
    if name == "HuggingFaceArtModelWrapper":
        from .declarations import HuggingFaceArtModelWrapper

        return HuggingFaceArtModelWrapper
    if name == "HuggingFacePytorchModelConfig":
        from .model import HuggingFacePytorchModelConfig

        return HuggingFacePytorchModelConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
