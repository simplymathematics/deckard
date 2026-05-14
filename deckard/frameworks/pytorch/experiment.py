"""PyTorch-specific experiment orchestration for deckard.

Provides TorchExperimentConfig, a PyTorch-specific orchestration layer that
enforces PyTorch backend consistency across all components: data, model, attack,
and device handling. The library parameter is hardcoded to "pytorch", and all
components must be PyTorch-compatible subclasses of the base config objects.
Supports ART PyTorchClassifier and PyTorchRegressor wrappers for defenses.
"""

import logging
from typing import Any, Union

from ...experiment.base import ExperimentConfig
from ...utils import is_default_config_value, is_null_config_value, resolve_torch_device

try:
    from .data import PytorchDataConfig
except ImportError:  # pragma: no cover
    PytorchDataConfig = None

try:
    from .model import PytorchModelConfig
except ImportError:  # pragma: no cover
    PytorchModelConfig = None

logger = logging.getLogger(__name__)


class TorchExperimentConfig(ExperimentConfig):
    """Experiment configuration for PyTorch models.

    Enforces:

    * ``data`` is a :class:`~deckard.frameworks.pytorch.data.PytorchDataConfig` instance
    * ``model`` is a :class:`~deckard.frameworks.pytorch.model.PytorchModelConfig` instance
    * ``library`` is always ``"pytorch"`` (serialization compatibility field only)

    Device reconciliation and PyTorch-specific device-setting logic live here
    rather than in the base :class:`ExperimentConfig`.
    """

    # Hardcode library; keep field so YAML round-trips work, but assert in __post_init__.
    library: str = "pytorch"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_device(device_value: Any) -> Union[str, None]:
        if is_null_config_value(
            device_value,
            allow_empty=True,
        ) or is_default_config_value(
            device_value,
            include_best=True,
        ):
            return None
        if device_value is None:
            return None
        text = str(device_value).strip()
        return text.lower() if text else None

    def _reconcile_component_devices(self) -> None:
        """Ensure all PyTorch components share a single device."""
        exp_device = self._canonical_device(getattr(self, "device", None))
        data_device = self._canonical_device(getattr(self.data, "device", None))
        model_device = (
            self._canonical_device(getattr(self.model, "device", None))
            if self.model is not None
            else None
        )
        attack_device = (
            self._canonical_device(getattr(self.attack, "device", None))
            if self.attack is not None
            else None
        )

        raw_values = [exp_device, data_device, model_device, attack_device]
        strong_values = {v for v in raw_values if v not in {None, "", "cpu"}}
        if len(strong_values) > 1:
            raise AssertionError(
                "Experiment, data, model, and attack devices must match. "
                f"Got experiment={exp_device}, data={data_device}, "
                f"model={model_device}, attack={attack_device}",
            )

        if exp_device is not None:
            unified_device = exp_device
        elif len(strong_values) == 1:
            unified_device = next(iter(strong_values))
        elif model_device is not None:
            unified_device = model_device
        elif attack_device is not None:
            unified_device = attack_device
        elif data_device is not None:
            unified_device = data_device
        else:
            unified_device = str(resolve_torch_device(None))

        self.device = unified_device
        setattr(self.data, "device", unified_device)
        if self.model is not None:
            setattr(self.model, "device", unified_device)
            if hasattr(self.model, "_resolve_torch_device") and callable(
                getattr(self.model, "_resolve_torch_device"),
            ):
                self.model.device = self.model._resolve_torch_device(
                    self.model.device,
                )
        if self.attack is not None:
            setattr(self.attack, "device", unified_device)

        final_exp = self._canonical_device(self.device)
        final_data = self._canonical_device(getattr(self.data, "device", None))
        final_model = (
            self._canonical_device(getattr(self.model, "device", None))
            if self.model is not None
            else final_exp
        )
        final_attack = (
            self._canonical_device(getattr(self.attack, "device", None))
            if self.attack is not None
            else final_exp
        )
        if not (final_exp == final_data == final_model == final_attack):
            raise AssertionError(
                "Experiment, data, model, and attack devices must be identical after "
                "reconciliation. "
                f"Got experiment={final_exp}, data={final_data}, "
                f"model={final_model}, attack={final_attack}",
            )
        logger.info("Unified pytorch device across components: %s", final_exp)

    def _reconcile_batch_size(self) -> None:
        """Keep model/data batch_size aligned for torch experiments.

        Rules:
        - If both are defined and differ: raise ValueError.
        - If only one side is defined: copy it to the other side.
        - If neither is defined: leave unchanged.
        """
        if self.model is None:
            return

        fit_params = getattr(self.model, "fit_params", None)
        if fit_params is None:
            fit_params = {}
            self.model.fit_params = fit_params

        data_params = getattr(self.data, "data_params", None)
        if data_params is None:
            data_params = {}
            self.data.data_params = data_params

        model_batch_size = fit_params.get("batch_size")
        data_batch_size = data_params.get("batch_size")

        if model_batch_size is not None and data_batch_size is not None:
            if int(model_batch_size) != int(data_batch_size):
                raise ValueError(
                    "TorchExperimentConfig requires model.fit_params.batch_size "
                    "and data.data_params.batch_size to match when both are set. "
                    f"Got model={model_batch_size}, data={data_batch_size}.",
                )
            return

        if model_batch_size is not None:
            data_params["batch_size"] = int(model_batch_size)
            logger.info(
                "Set data.data_params.batch_size from model.fit_params.batch_size: %s",
                model_batch_size,
            )
            return

        if data_batch_size is not None:
            fit_params["batch_size"] = int(data_batch_size)
            logger.info(
                "Set model.fit_params.batch_size from data.data_params.batch_size: %s",
                data_batch_size,
            )
            return

    def set_device(self, device: Union[str, int] = "auto") -> None:
        """Configure the PyTorch device for this experiment."""
        torch_device = resolve_torch_device(device)
        self.torch_device = torch_device
        self.device = str(torch_device)

    # ------------------------------------------------------------------
    # Type-enforcement helpers
    # ------------------------------------------------------------------

    def _enforce_torch_data(self) -> None:
        """Raise TypeError when data is not a PytorchDataConfig."""
        if PytorchDataConfig is None:
            raise ImportError(
                "TorchExperimentConfig requires the optional pytorch data dependency. "
                "Install deckard[torch] to enable it.",
            )

        if not isinstance(self.data, PytorchDataConfig):
            raise TypeError(
                f"TorchExperimentConfig requires data to be a PytorchDataConfig "
                f"(or subclass), but got {type(self.data).__name__}. "
                "Use PytorchDataConfig for torch experiments.",
            )

    def _enforce_torch_model(self) -> None:
        """Raise TypeError when model is not a PytorchModelConfig."""
        if self.model is None:
            return  # model-less experiments are allowed

        if PytorchModelConfig is None:
            raise ImportError(
                "TorchExperimentConfig requires the optional pytorch model dependency. "
                "Install deckard[torch] to enable it.",
            )

        if not isinstance(self.model, PytorchModelConfig):
            raise TypeError(
                f"TorchExperimentConfig requires model to be a PytorchModelConfig "
                f"(or subclass), but got {type(self.model).__name__}. "
                "Use PytorchModelConfig for torch experiments.",
            )

    # ------------------------------------------------------------------
    # __post_init__
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.library != "pytorch":
            raise ValueError(
                f"TorchExperimentConfig must use library='pytorch', got {self.library!r}.",
            )

        # Let base class wire together data / model / attack / files / score.
        super().__post_init__()

        # After base wiring, enforce torch types.
        self._enforce_torch_data()
        self._enforce_torch_model()
        self._reconcile_component_devices()
        self._reconcile_batch_size()
