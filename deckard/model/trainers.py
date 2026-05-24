"""Pluggable trainer objects for model runtime orchestration.

These trainer objects mirror the resolve/compose/execute pattern used by
DataConfig samplers, while keeping ModelConfig as the owner of persistence,
scoring, and defense orchestration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sklearn.exceptions import NotFittedError

from .canon import normalize_model_trainer_alias
from ..utils import load_class

if TYPE_CHECKING:
    from .base import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class BaseTrainer:
    """Trainer interface plus centralized trainer composition helpers."""

    def __call__(
        self,
        config: "ModelConfig",
        data: Any,
        *,
        model_file: str | None = None,
        times: dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Train model runtime and return training metadata.

        Args:
            config: Runtime model config.
            data: Runtime data config containing train payloads.
            model_file: Optional model artifact path.
            times: Optional runtime timing mapping.
            force_retrain: Whether to force retraining even when artifacts exist.

        Returns:
            Training metadata payload.

        Raises:
            NotImplementedError: Always raised by the base trainer interface.
        """
        raise NotImplementedError

    @classmethod
    def resolve(cls, config: "ModelConfig") -> Any:
        """Resolve config.trainer into a callable trainer object or None.

        Args:
            config: Runtime model config with trainer declaration.

        Returns:
            Resolved trainer object or None.

        Raises:
            ValueError: If trainer specification type is unsupported.
        """
        trainer_aliases = {
            "sklearn": SklearnTrainer,
            "pretrained": PretrainedTrainer,
            "partial_fit": PartialFitTrainer,
            "partial_fit_pruning": PartialFitPruningTrainer,
            "pruning": PruningTrainer,
            "pytorch": PytorchTrainer,
        }

        spec = getattr(config, "trainer", None)
        if spec is None:
            return None

        if isinstance(spec, str):
            key = normalize_model_trainer_alias(spec)
            trainer_cls = trainer_aliases[key]
            params = dict(getattr(config, "trainer_params", {}) or {})
            return trainer_cls(**params)

        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(spec, DictConfig):
                spec = OmegaConf.to_container(spec, resolve=True)
        except ImportError:
            pass

        if isinstance(spec, dict):
            if not spec:
                return None
            payload = dict(spec)
            class_path = payload.pop("name", payload.pop("_target_", None))
            if class_path is None:
                raise ValueError("trainer dict must include 'name' or '_target_'")
            return load_class(class_path, **payload)

        if callable(spec) and not isinstance(spec, type):
            return spec

        if isinstance(spec, type):
            return spec()

        raise ValueError(f"Unsupported trainer specification: {type(spec)}")

    @classmethod
    def compose(cls, config: "ModelConfig") -> Any:
        """Compose and cache runtime trainer callable for config.

        Args:
            config: Runtime model config.

        Returns:
            Callable trainer object.

        Raises:
            TypeError: If composed trainer is not callable.
        """
        trainer_obj = getattr(config, "_trainer_obj", None)
        if trainer_obj is None:
            trainer_obj = cls.resolve(config)
            setattr(config, "_trainer_obj", trainer_obj)
        if trainer_obj is None:
            trainer_obj = SklearnTrainer()
            setattr(config, "_trainer_obj", trainer_obj)
        if not callable(trainer_obj):
            raise TypeError(f"Composed trainer must be callable, got {type(trainer_obj)}")
        return trainer_obj

    @classmethod
    def execute(
        cls,
        config: "ModelConfig",
        data: Any,
        *,
        model_file: str | None = None,
        times: dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Resolve/compose and run the configured trainer against config.

        Args:
            config: Runtime model config.
            data: Runtime data config.
            model_file: Optional model artifact path.
            times: Optional runtime timing mapping.
            force_retrain: Whether to force retraining.

        Returns:
            Training metadata payload from trainer execution.
        """
        trainer_obj = cls.compose(config)
        return trainer_obj(
            config,
            data,
            model_file=model_file,
            times=times,
            force_retrain=force_retrain,
        )


@dataclass
class SklearnTrainer(BaseTrainer):
    """Default trainer for sklearn-style fit workflows."""

    def __call__(
        self,
        config: "ModelConfig",
        data: Any,
        *,
        model_file: str | None = None,
        times: dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Train sklearn-style model and optionally persist artifact.

        Args:
            config: Runtime model config.
            data: Runtime data config.
            model_file: Optional model artifact path.
            times: Optional runtime timing mapping.
            force_retrain: Unused for sklearn trainer; retained for interface parity.

        Returns:
            Training metadata payload.
        """
        output = dict(times or {})
        config._train(data.X_train, data.y_train)
        output["training_time"] = getattr(config, "training_time", None)
        output["training_n"] = getattr(config, "training_n", None)
        if model_file is not None:
            config.save_object(config, model_file)
        return output


@dataclass
class PytorchTrainer(BaseTrainer):
    """Trainer for torch model flows; delegates to model _train implementation."""

    def __call__(
        self,
        config: "ModelConfig",
        data: Any,
        *,
        model_file: str | None = None,
        times: dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Train torch-style model and optionally persist artifact.

        Args:
            config: Runtime model config.
            data: Runtime data config.
            model_file: Optional model artifact path.
            times: Optional runtime timing mapping.
            force_retrain: Unused for pytorch trainer; retained for interface parity.

        Returns:
            Training metadata payload.
        """
        output = dict(times or {})
        config._train(data.X_train, data.y_train)
        output["training_time"] = getattr(config, "training_time", None)
        output["training_n"] = getattr(config, "training_n", None)
        if model_file is not None:
            config.save_object(config, model_file)
        return output


@dataclass
class PretrainedTrainer(BaseTrainer):
    """Trainer policy that prefers pre-trained model artifacts and avoids retraining."""

    allow_fallback_training: bool = False

    def __call__(
        self,
        config: "ModelConfig",
        data: Any,
        *,
        model_file: str | None = None,
        times: dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Load pretrained artifacts when available and optionally retrain.

        Args:
            config: Runtime model config.
            data: Runtime data config.
            model_file: Optional model artifact path.
            times: Optional runtime timing mapping.
            force_retrain: Whether to bypass artifact loading and retrain.

        Returns:
            Training metadata payload.

        Raises:
            NotFittedError: If no fitted artifact is available and fallback training is disabled.
        """
        output = dict(times or {})
        if force_retrain:
            config._initialize_model()
            config._train(data.X_train, data.y_train)
            output["training_time"] = getattr(config, "training_time", None)
            output["training_n"] = getattr(config, "training_n", None)
            if model_file is not None:
                config.save_object(config, model_file)
            return output

        if model_file is not None and Path(model_file).exists():
            loaded_obj = config.load(str(model_file))
            if hasattr(loaded_obj, "_model"):
                config.__dict__.update(getattr(loaded_obj, "__dict__", {}))
            else:
                config._model = loaded_obj
            if config._is_model_fitted(config._model, X_sample=data.X_train):
                logger.info("Pretrained trainer loaded fitted model from %s", model_file)
                output.setdefault("training_time", getattr(config, "training_time", None))
                output.setdefault("training_n", getattr(config, "training_n", None))
                return output

        if self.allow_fallback_training:
            config._train(data.X_train, data.y_train)
            output["training_time"] = getattr(config, "training_time", None)
            output["training_n"] = getattr(config, "training_n", None)
            if model_file is not None:
                config.save_object(config, model_file)
            return output

        raise NotFittedError(
            "PretrainedTrainer requires a fitted model artifact and allow_fallback_training=False",
        )


@dataclass
class PartialFitTrainer(BaseTrainer):
    """Incremental trainer using partial_fit when available."""

    classes: list[Any] | None = None

    def __call__(
        self,
        config: "ModelConfig",
        data: Any,
        *,
        model_file: str | None = None,
        times: dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Incrementally train model with partial_fit when supported.

        Args:
            config: Runtime model config.
            data: Runtime data config.
            model_file: Optional model artifact path.
            times: Optional runtime timing mapping.
            force_retrain: Unused for partial-fit trainer; retained for interface parity.

        Returns:
            Training metadata payload.
        """
        output = dict(times or {})
        model = getattr(config, "_model", None)
        if model is None and hasattr(config, "_initialize_model"):
            config._initialize_model()
            model = getattr(config, "_model", None)

        partial_fit = getattr(model, "partial_fit", None)
        if not callable(partial_fit):
            config._train(data.X_train, data.y_train)
        else:
            fit_params = getattr(config, "fit_params", {}) or {}
            if getattr(config, "classifier", True):
                classes = self.classes
                if classes is None:
                    try:
                        import numpy as np

                        classes = np.unique(data.y_train)
                    except Exception:
                        classes = None
                if classes is not None and "classes" not in fit_params:
                    fit_params = {**fit_params, "classes": classes}
            partial_fit(data.X_train, data.y_train, **fit_params)
            config.training_n = len(data.y_train)

        output["training_time"] = getattr(config, "training_time", None)
        output["training_n"] = getattr(config, "training_n", None)
        if model_file is not None:
            config.save_object(config, model_file)
        return output


@dataclass
class PruningTrainer(BaseTrainer):
    """Trainer with optional trial pruning hook support."""

    trial: Any = None
    prune_metric: str = "training_time"
    prune_step: int = 0

    def __call__(
        self,
        config: "ModelConfig",
        data: Any,
        *,
        model_file: str | None = None,
        times: dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Train model and optionally mark trial as pruned using configured metric.

        Args:
            config: Runtime model config.
            data: Runtime data config.
            model_file: Optional model artifact path.
            times: Optional runtime timing mapping.
            force_retrain: Unused for pruning trainer; retained for interface parity.

        Returns:
            Training metadata payload.
        """
        output = dict(times or {})
        config._train(data.X_train, data.y_train)
        output["training_time"] = getattr(config, "training_time", None)
        output["training_n"] = getattr(config, "training_n", None)

        if self.trial is not None and hasattr(config, "check_prune"):
            metric_value = output.get(self.prune_metric, None)
            should_prune = config.check_prune(
                self.trial,
                value=metric_value,
                step=self.prune_step,
            )
            if should_prune:
                output["pruned"] = True

        if model_file is not None:
            config.save_object(config, model_file)
        return output


@dataclass
class PartialFitPruningTrainer(PartialFitTrainer):
    """Incremental partial-fit trainer with optional pruning checks."""

    trial: Any = None
    prune_metric: str = "training_time"
    prune_step: int = 0

    def __call__(
        self,
        config: "ModelConfig",
        data: Any,
        *,
        model_file: str | None = None,
        times: dict[str, Any] | None = None,
        force_retrain: bool = False,
    ) -> dict[str, Any]:
        """Run partial-fit training with optional pruning checks.

        Args:
            config: Runtime model config.
            data: Runtime data config.
            model_file: Optional model artifact path.
            times: Optional runtime timing mapping.
            force_retrain: Unused for this trainer; retained for interface parity.

        Returns:
            Training metadata payload.
        """
        output = super().__call__(
            config,
            data,
            model_file=model_file,
            times=times,
            force_retrain=force_retrain,
        )
        if self.trial is not None and hasattr(config, "check_prune"):
            metric_value = output.get(self.prune_metric, None)
            should_prune = config.check_prune(
                self.trial,
                value=metric_value,
                step=self.prune_step,
            )
            if should_prune:
                output["pruned"] = True
        return output


__all__ = [
    "BaseTrainer",
    "SklearnTrainer",
    "PretrainedTrainer",
    "PartialFitTrainer",
    "PruningTrainer",
    "PartialFitPruningTrainer",
    "PytorchTrainer",
]
