"""HuggingFace-aware ART model config for the Deckard transformers framework.

Provides :class:`HuggingFacePytorchModelConfig`, which extends the standard
:class:`~deckard.frameworks.pytorch.model.PytorchModelConfig` and overrides
:meth:`get_art_model` to use ART's
:class:`~art.estimators.classification.HuggingFaceClassifierPyTorch` instead of
the generic :class:`~art.estimators.classification.PyTorchClassifier`.

This avoids the float32 cast that :class:`~art.estimators.classification.PyTorchClassifier`
applies to all inputs, which would corrupt integer ``input_ids`` required by
HuggingFace embedding layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
except Exception:  # pragma: no cover - optional dependency import may fail at runtime
    torch = None
    DataLoader = None
    Dataset = None
    Subset = None

try:
    from art.estimators.classification import HuggingFaceClassifierPyTorch
except ImportError:  # pragma: no cover
    HuggingFaceClassifierPyTorch = None

from ..pytorch.model import (
    PytorchModelConfig,
    initialize_criterion,
    initialize_optimizer,
)

__all__ = ["HuggingFacePytorchModelConfig"]

if TYPE_CHECKING:
    from ...data import DataConfig


if HuggingFaceClassifierPyTorch is not None:

    class DeckardHuggingFaceClassifierPyTorch(HuggingFaceClassifierPyTorch):
        """ART HF classifier with MPS-aware hook input placement.

        ART's private `_make_model_wrapper()` builds a dummy input for layer discovery
        and only moves it for CUDA models. On MPS this leaves the hook input on CPU,
        which fails before Deckard can override ART's device internals.
        """

        def _make_model_wrapper(self, model):
            import torch

            input_shape = self._input_shape
            model_device = next(model.parameters()).device
            input_for_hook = torch.rand(input_shape, device=model_device)
            input_for_hook = torch.unsqueeze(input_for_hook, dim=0)

            if self.processor is not None:
                input_for_hook = self.processor(input_for_hook)

            processor = self.processor

            if not hasattr(self, "_model_wrapper"):

                class ModelWrapper(torch.nn.Module):
                    def __init__(self, model: torch.nn.Module):
                        super().__init__()
                        self._model = model

                    def forward(self, x):
                        result = []

                        if isinstance(self._model, torch.nn.Module):
                            if processor is not None:
                                x = processor(x)
                            x = self._model.forward(x)
                            result.append(x)
                        else:  # pragma: no cover
                            raise TypeError(
                                "The input model must inherit from `nn.Module`.",
                            )

                        return result

                    @property
                    def get_layers(self) -> list[str]:
                        result_dict = {}
                        modules = []

                        def forward_hook(input_module, hook_input, hook_output):
                            _ = hook_input
                            _ = hook_output
                            modules.append(id(input_module))

                        handles = []

                        for name, module in self._model.named_modules():
                            if name != "" and len(list(module.named_modules())) == 1:
                                handles.append(
                                    module.register_forward_hook(forward_hook),
                                )
                                result_dict[id(module)] = name

                        model(input_for_hook)

                        for hook in handles:
                            hook.remove()

                        return [result_dict[module] for module in modules]

                self._model_wrapper = ModelWrapper

            return self._model_wrapper(model)

else:  # pragma: no cover
    DeckardHuggingFaceClassifierPyTorch = None


@dataclass(eq=False, kw_only=True)
class HuggingFacePytorchModelConfig(PytorchModelConfig):
    """PytorchModelConfig that wires ART's HuggingFaceClassifierPyTorch.

    Use this config (``_target_: deckard.frameworks.transformers.model.HuggingFacePytorchModelConfig``)
    instead of the base ``PytorchModelConfig`` when the model is a HuggingFace
    transformer (e.g. :class:`~deckard.frameworks.transformers.declarations.GenericFlexibleTransformer`).
    """

    def get_art_model(self, data: "DataConfig"):
        """Build an ART HuggingFaceClassifierPyTorch around the inner torch model.

        The inner model is wrapped in :class:`~deckard.frameworks.transformers.declarations.HuggingFaceArtModelWrapper`
        which (a) casts float input tensors back to ``long`` and (b) returns an
        object with a ``.logits`` attribute as required by ART's HF classifier.

        Args:
            data: Runtime data config for shape / class-count inference.

        Returns:
            A configured :class:`~art.estimators.classification.HuggingFaceClassifierPyTorch` instance.
        """
        if DeckardHuggingFaceClassifierPyTorch is None:
            raise ImportError(
                "HuggingFacePytorchModelConfig requires "
                "adversarial-robustness-toolbox >= 1.18 with HuggingFace support.",
            )

        from .declarations import HuggingFaceArtModelWrapper

        clip_values = tuple(self.clip_values) if self.clip_values else (0.0, 1.0)

        batch_size = getattr(data, "batch_size", None) or self.fit_params.get(
            "batch_size",
            32,
        )

        if isinstance(data.X_train, torch.utils.data.DataLoader):
            loader = data.X_train
        elif isinstance(data.X_train, (Dataset, Subset)):
            loader = DataLoader(data.X_train, batch_size=batch_size, shuffle=False)
        else:
            loader = None

        if loader is not None:
            batch = next(iter(loader))
            if isinstance(batch, (tuple, list)):
                input_shape = tuple(batch[0].shape[1:])
            else:
                input_shape = tuple(batch.shape[1:])
        else:
            input_shape = tuple(data.X_train.shape[1:])

        nb_classes = int(len(torch.unique(data.y_train)))
        if nb_classes < 2:
            configured_classes = int(getattr(self, "num_classes", 0) or 0)
            nb_classes = configured_classes if configured_classes >= 2 else 2
        art_device_type = self._resolve_art_device_type()

        wrapped = HuggingFaceArtModelWrapper(self._model)
        estimator = DeckardHuggingFaceClassifierPyTorch(
            model=wrapped,
            loss=initialize_criterion(self.criterion),
            optimizer=initialize_optimizer(
                self.optimizer,
                self._model.parameters(),
            ),
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values,
            device_type=art_device_type,
        )
        return self._override_art_internal_device(estimator)
