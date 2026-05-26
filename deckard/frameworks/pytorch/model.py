from __future__ import annotations

# OS imports
import copy
import inspect
import logging
import time

# Typing imports
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Union

import numpy as np
from omegaconf import DictConfig

# Torch imports (optional dependency)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader as TorchDataLoader
except ImportError:
    torch = None
    nn = None
    TorchDataLoader = None


# Sklearn imports
# ART imports
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from ...data.base import DataConfig
from ..types import EstimatorLike
from ...model.base import ModelConfig
from ...utils import is_default_config_value, load_class, resolve_torch_device

logger = logging.getLogger(__name__)

ScorerDictConfig = Any
ModelType = Union[str, type[torch.nn.Module], torch.nn.Module]

__all__ = ["PytorchModelConfig"]


class RuntimeAttackPayload(Protocol):
    """Opaque marker protocol for runtime epoch-attack configuration payloads."""


# TinyNet: Minimal torch model for binary classification
class TinyNet(nn.Module if nn else object):
    """A minimal torch model for binary classification (2-layer MLP)."""

    def __init__(self, input_dim=10, hidden_dim=16, output_dim=2):
        if nn is None:
            raise ImportError("TinyNet requires torch to be installed.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Run forward pass through TinyNet.

        Args:
            x: Input tensor batch.

        Returns:
            Logit tensor output.
        """
        return self.net(x)


def initialize_criterion(criterion_spec):
    """Initialize criterion from string or dict using load_class."""
    if isinstance(criterion_spec, str):
        if "." not in criterion_spec and ":" not in criterion_spec:
            criterion_spec = f"torch.nn.{criterion_spec}"
        return load_class(criterion_spec)
    elif isinstance(criterion_spec, (dict, DictConfig)):
        criterion_name = criterion_spec.get("name") or criterion_spec.get(
            "_target_",
        )
        criterion_params = {
            k: v for k, v in criterion_spec.items() if k not in ["name", "_target_"]
        }
        return load_class(criterion_name, **criterion_params)
    else:
        raise ValueError(
            f"criterion must be str or dict, got {type(criterion_spec)}",
        )


def initialize_optimizer(optimizer_spec, model_params):
    """Initialize optimizer from string or dict using load_class."""
    if isinstance(optimizer_spec, str):
        if "." not in optimizer_spec and ":" not in optimizer_spec:
            optimizer_spec = f"torch.optim.{optimizer_spec}"
        return load_class(optimizer_spec, model_params)
    elif isinstance(optimizer_spec, (dict, DictConfig)):
        optimizer_name = optimizer_spec.get("name") or optimizer_spec.get(
            "_target_",
        )
        if (
            isinstance(optimizer_name, str)
            and "." not in optimizer_name
            and ":" not in optimizer_name
        ):
            optimizer_name = f"torch.optim.{optimizer_name}"
        optimizer_params = {
            k: v for k, v in optimizer_spec.items() if k not in ["name", "_target_"]
        }
        optimizer_params["params"] = model_params
        return load_class(optimizer_name, **optimizer_params)
    else:
        raise ValueError(
            f"optimizer must be str or dict, got {type(optimizer_spec)}",
        )


@dataclass(eq=False, kw_only=True)
class PytorchModelConfig(ModelConfig):
    """Configuration for PyTorch models using load_class for generic instantiation.

    Attributes:
        model_type: Fully qualified class path, an in-memory class, or an nn.Module instance
        model_params: Constructor parameters for the model
        device: torch.device ("cpu", "cuda", etc.)
        criterion: Loss function spec (str name or dict with _target_)
        optimizer: Optimizer spec (str name or dict with _target_)
        fit_params: Parameters for model training
        classifier: Whether model is classifier (True) or regressor (False)
    """

    model_type: ModelType = "torch.nn.Linear"
    model_params: dict = field(default_factory=dict)
    classifier: bool = True
    fit_params: dict = field(default_factory=dict)
    library: str = "pytorch"
    device: Any = None
    criterion: Any = field(default="torch.nn.CrossEntropyLoss")
    optimizer: Any = field(default="torch.optim.SGD")
    clip_values: Union[tuple, None] = None
    random_seed: int = 42
    channels_first: bool = True
    checkpoint_records: list = field(default_factory=list)
    _checkpoint_context: Any = field(default=None, repr=False, compare=False)
    _epoch_attack: Any = field(default=None, repr=False, compare=False)

    @staticmethod
    def _pickle_safe_model_type(model_type: Any) -> Any:
        if model_type is None or isinstance(model_type, str):
            return model_type
        if isinstance(model_type, type):
            return f"{model_type.__module__}.{model_type.__qualname__}"
        return f"{model_type.__class__.__module__}.{model_type.__class__.__qualname__}"

    def __getstate__(self):
        state = dict(self.__dict__)
        model_obj = state.get("_model", None)
        if model_obj is not None and hasattr(model_obj, "state_dict"):
            try:
                state["_pickled_model_state_dict"] = copy.deepcopy(
                    model_obj.state_dict(),
                )
            except Exception:
                state["_pickled_model_state_dict"] = None
            state["_model"] = None

        state["model_type"] = self._pickle_safe_model_type(state.get("model_type"))
        state["_checkpoint_context"] = None
        state["_epoch_attack"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        pickled_state = self.__dict__.pop("_pickled_model_state_dict", None)
        if getattr(self, "_model", None) is None:
            try:
                self._initialize_model()
            except Exception:
                return
        if pickled_state is not None and getattr(self, "_model", None) is not None:
            try:
                self._model.load_state_dict(pickled_state)
                self._model = self._model.to(self.device)
            except Exception:
                pass

    def _initialize_default_scorer(self) -> None:
        if not is_default_config_value(self.scorer, include_best=False):
            return
        scorer_cls = (
            "deckard.score.base.DefaultPytorchClassifierScorerDictConfig"
            if self.classifier
            else "deckard.score.base.DefaultPytorchRegressorScorerDictConfig"
        )
        self.scorer = load_class(scorer_cls)

    def _initialize_torch_seed_and_device(self) -> None:
        self.device = self._resolve_torch_device(self.device)

        torch.manual_seed(self.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)

    def __post_init__(self):
        """Initialize model using load_class, following parent pattern."""
        self._initialize_default_scorer()
        self._initialize_torch_seed_and_device()

        # For in-memory model instances, infer constructor params so config
        # metadata remains serializable and reproducible.
        if isinstance(self.model_type, torch.nn.Module):
            inferred_params = self._infer_model_init_params_from_instance(
                self.model_type,
            )
            if self.model_params is None:
                self.model_params = inferred_params
            else:
                merged_params = dict(inferred_params)
                merged_params.update(dict(self.model_params))
                self.model_params = merged_params

        # Call parent __post_init__ for shared initialization
        super().__post_init__()

    def _infer_model_init_params_from_instance(
        self,
        model_instance: torch.nn.Module,
    ) -> dict:
        """Best-effort extraction of __init__ args from an in-memory model."""
        try:
            signature = inspect.signature(model_instance.__class__.__init__)
        except (TypeError, ValueError):
            return {}

        inferred = {}
        first_param = next(model_instance.parameters(), None)
        for param in signature.parameters.values():
            if param.name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            name = param.name
            if name == "bias" and hasattr(model_instance, "bias"):
                inferred[name] = getattr(model_instance, "bias") is not None
                continue

            if hasattr(model_instance, name):
                value = getattr(model_instance, name)
                if isinstance(value, torch.nn.Parameter):
                    # Constructor args usually expect bool for bias, not Parameter.
                    inferred[name] = value is not None
                elif isinstance(value, torch.Tensor):
                    continue
                else:
                    inferred[name] = copy.deepcopy(value)
                continue

            if name == "device" and first_param is not None:
                inferred[name] = first_param.device
                continue

            if name == "dtype" and first_param is not None:
                inferred[name] = first_param.dtype
                continue

        return inferred

    def _resolve_torch_device(self, requested_device: Any) -> torch.device:
        return resolve_torch_device(requested_device)

    def _resolve_art_device_type(self) -> str:
        if self.device.type == "cuda":
            return "gpu"
        if self.device.type == "mps":
            logger.info(
                "ART device_type has no native MPS option; overriding ART internal device directly",
            )
            return "cpu"
        return "cpu"

    def _model_for_art(self):
        if self._model is None:
            raise ValueError("Model not initialized")
        if isinstance(self._model, (PyTorchClassifier, PyTorchRegressor)):
            return self._model
        if self.device.type == "mps":
            # Keep training model on MPS while giving ART a CPU-compatible copy.
            model_copy = copy.deepcopy(self._model)
            return model_copy.to(torch.device("cpu"))
        return self._model

    def _override_art_internal_device(self, art_estimator):
        if self.device.type not in {"mps", "cuda", "cpu"}:
            return art_estimator

        target_device = self.device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            target_device = torch.device("cpu")
        if self.device.type == "mps":
            # ART preprocessors often materialize float64 tensors, which MPS cannot run.
            # Keep the wrapped ART estimator on CPU while the training model itself can still
            # use MPS through the separate copy returned by _model_for_art().
            target_device = torch.device("cpu")

        if hasattr(art_estimator, "_device"):
            art_estimator._device = target_device
        if hasattr(art_estimator, "_model") and hasattr(
            art_estimator._model,
            "to",
        ):
            art_estimator._model = art_estimator._model.to(target_device)

        preprocessing = getattr(art_estimator, "preprocessing", None)
        if hasattr(preprocessing, "_device"):
            preprocessing._device = target_device

        for op in getattr(art_estimator, "preprocessing_operations", []) or []:
            if hasattr(op, "_device"):
                op._device = target_device

        # Some ART preprocessors (e.g., FeatureSqueezing) can promote arrays to float64.
        # Force ART preprocessing outputs back to ART_NUMPY_DTYPE before torch forward.
        if not getattr(art_estimator, "_deckard_dtype_wrapped", False):
            original_apply_preprocessing = getattr(
                art_estimator,
                "_apply_preprocessing",
                None,
            )
            if callable(original_apply_preprocessing):

                def _dtype_safe_apply_preprocessing(*args, **kwargs):
                    result = original_apply_preprocessing(*args, **kwargs)
                    if isinstance(result, tuple) and len(result) >= 1:
                        x_out = result[0]
                        if isinstance(x_out, np.ndarray) and np.issubdtype(
                            x_out.dtype,
                            np.floating,
                        ):
                            x_out = x_out.astype(ART_NUMPY_DTYPE, copy=False)
                        if len(result) == 1:
                            return (x_out,)
                        return (x_out, *result[1:])
                    return result

                art_estimator._apply_preprocessing = _dtype_safe_apply_preprocessing
                art_estimator._deckard_dtype_wrapped = True

        return art_estimator

    def set_epoch_attack(self, attack_config: RuntimeAttackPayload) -> None:
        """Attach an optional per-epoch adversarial attack configuration.

        Args:
            attack_config: Runtime attack configuration payload used during training.

        Returns:
            None.
        """
        self._epoch_attack = attack_config

    def _initialize_model(self):
        """Initialize PyTorch model from path, class, or in-memory instance."""
        params = self.model_params if self.model_params is not None else {}
        if isinstance(self.model_type, torch.nn.Module):
            # Avoid mutating a caller-owned module instance.
            self._model = copy.deepcopy(self.model_type)
        elif isinstance(self.model_type, type):
            if not issubclass(self.model_type, torch.nn.Module):
                raise TypeError(
                    "model_type class must inherit torch.nn.Module",
                )
            self._model = self.model_type(**params)
        elif self.model_params is not None:
            self._model = load_class(self.model_type, **self.model_params)
        else:
            self._model = load_class(self.model_type)

        # Move model to device
        self._model = self._model.to(self.device)
        logger.info(
            f"Initialized model {self.model_type} on device {self.device}",
        )

    def get_model(self) -> ModelType:
        """Return the underlying PyTorch model.

        Returns:
            The initialized torch model instance.

        Raises:
            ValueError: If model is not initialized.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        return self._model

    def save(self, filepath: str) -> None:
        """Persist config state as canonical YAML artifact.

        Args:
            filepath: Output path for the YAML config state.

        Returns:
            None.
        """
        super().save(self, filepath)

    def load(self, filepath: str) -> "PytorchModelConfig":
        """Load config state from canonical YAML artifact.

        Args:
            filepath: Input path for the YAML config state.

        Returns:
            The current model config instance with restored model state.
        """
        loaded = super().load(filepath)
        self._initialize_model()
        return loaded

    def save_model(
        self,
        model: torch.nn.Module | str | Path | None = None,
        filepath: str | None = None,
        *,
        model_file: str | None = None,
    ) -> None:
        """Persist runtime torch model state to .pt or pickle payload.

        Args:
            model: Runtime model to serialize; defaults to internal model when None.
                Backward-compatible shorthand also accepts a path value.
            filepath: Output artifact path.
            model_file: Optional alias for ``filepath``.

        Raises:
            ValueError: If filepath suffix is unsupported or model is missing.
        """
        target_path = filepath if filepath is not None else model_file
        if target_path is None and isinstance(model, (str, Path)):
            target_path = str(model)
            model = None

        if target_path is None or str(target_path).strip() == "":
            # Match base ModelConfig.save_model semantics: persistence is optional
            # and callers may omit model artifact paths.
            return

        path = Path(target_path)
        suffix = path.suffix.lower()
        if suffix not in {".pt", ".pkl", ".pickle"}:
            raise ValueError(
                f"PytorchModelConfig runtime model artifacts must use .pt/.pkl/.pickle. Got: {suffix}",
            )

        model_obj = model if isinstance(model, torch.nn.Module) else self._model
        if model_obj is None:
            raise ValueError("Model not initialized")

        payload = {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "state_dict": model_obj.state_dict(),
            "device": str(self.device),
        }
        if suffix == ".pt":
            torch.save(payload, str(path))
        else:
            self.save_object(payload, str(path))

    def load_model(
        self,
        filepath: str,
        ignore_corrupt: bool = False,
        delete_corrupt: bool = False,
    ) -> ModelType:
        """Load runtime torch model state from .pt/.pkl payload.

        Args:
            filepath: Serialized model payload path.
            ignore_corrupt: Skip corrupt payload errors when supported.
            delete_corrupt: Delete corrupt payloads when supported.

        Returns:
            Restored torch model instance.

        Raises:
            TypeError: If serialized payload type is unsupported.
        """
        payload = super().load_model(
            filepath,
            ignore_corrupt=ignore_corrupt,
            delete_corrupt=delete_corrupt,
        )
        if isinstance(payload, dict) and "state_dict" in payload:
            self.model_type = payload.get("model_type", self.model_type)
            self.model_params = payload.get("model_params", self.model_params)
            self._initialize_model()
            self._model.load_state_dict(payload["state_dict"])
            self._model = self._model.to(self.device)
            return self._model
        if isinstance(payload, torch.nn.Module):
            self._model = payload.to(self.device)
            return self._model
        raise TypeError(f"Unsupported serialized torch model payload in {filepath}")

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _resolve_checkpoint_config(self, model_file=None):
        every = self.fit_params.get(
            "checkpoint_every_epochs",
            self.fit_params.get("checkpoint_every_cycles", None),
        )
        if every in {None, "", 0, "0"}:
            return None

        try:
            every = int(every)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint_every_epochs must be an integer, got {every}",
            ) from exc
        if every <= 0:
            return None

        checkpoint_dir = self.fit_params.get("checkpoint_dir", None)
        if checkpoint_dir is None:
            if model_file is None:
                raise ValueError(
                    "checkpoint_dir must be provided when checkpointing is enabled without a model_file",
                )
            model_path = Path(model_file)
            checkpoint_dir = model_path.parent / f"{model_path.stem}_checkpoints"
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_prefix = self.fit_params.get("checkpoint_prefix", None)
        if checkpoint_prefix in {None, ""}:
            if model_file is not None:
                checkpoint_prefix = Path(model_file).stem
            elif self.alias not in {None, ""}:
                checkpoint_prefix = self.alias
            else:
                checkpoint_prefix = str(self.model_type).split(":")[-1].split(".")[-1]

        return {
            "every": every,
            "dir": checkpoint_dir,
            "prefix": str(checkpoint_prefix),
            "score": self._coerce_bool(
                self.fit_params.get("checkpoint_score", True),
                default=True,
            ),
            "include_final": self._coerce_bool(
                self.fit_params.get("checkpoint_include_final", True),
                default=True,
            ),
        }

    def _build_checkpoint_snapshot(self):
        snapshot = type(self)(
            model_type=self.model_type,
            model_params=copy.deepcopy(self.model_params),
            classifier=self.classifier,
            fit_params=copy.deepcopy(self.fit_params),
            library=self.library,
            device=str(self.device),
            criterion=copy.deepcopy(self.criterion),
            optimizer=copy.deepcopy(self.optimizer),
            clip_values=copy.deepcopy(self.clip_values),
            random_seed=self.random_seed,
            channels_first=self.channels_first,
        )
        snapshot.alias = self.alias
        snapshot.scorer = self.scorer
        snapshot.defense = self.defense
        snapshot.score_dict = {}
        snapshot.checkpoint_records = copy.deepcopy(self.checkpoint_records)
        snapshot._initialize_model()
        snapshot._model.load_state_dict(copy.deepcopy(self._model.state_dict()))
        snapshot._model = snapshot._model.to(snapshot.device)
        return snapshot

    def _score_checkpoint_snapshot(self, snapshot, data):
        # Copy over epoch metrics from our score_dict if they exist
        if "epochs" in self.score_dict:
            snapshot.score_dict["epochs"] = copy.deepcopy(
                self.score_dict["epochs"],
            )
        # Expose the latest optimization loss directly in each checkpoint score
        # payload so downstream consumers do not need to parse nested epoch data.
        epochs = snapshot.score_dict.get("epochs", {}) if snapshot.score_dict else {}
        if isinstance(epochs, dict) and len(epochs) > 0:
            latest_epoch = None
            for key in epochs.keys():
                try:
                    epoch_index = int(key)
                except (TypeError, ValueError):
                    continue
                if latest_epoch is None or epoch_index > latest_epoch:
                    latest_epoch = epoch_index
            if latest_epoch is not None:
                latest_entry = epochs.get(latest_epoch, None)
                if latest_entry is None:
                    latest_entry = epochs.get(str(latest_epoch), None)
                if isinstance(latest_entry, dict):
                    latest_loss = latest_entry.get("loss", None)
                    if isinstance(latest_loss, (int, float)):
                        snapshot.score_dict["optimizer_loss"] = float(latest_loss)

        checkpoint_stage = None
        if snapshot.defense is not None:
            defense_pipeline = snapshot._require_defense_pipeline()
            checkpoint_stage = defense_pipeline.resolve_stage(
                default_stage="post_fit_pre_predict",
                model=snapshot,
                data=data,
            )
            if checkpoint_stage == "post_fit_pre_predict":
                snapshot._model = snapshot.apply_defense(data)
        try:
            snapshot._evaluate_and_score(data, times={})
        except ValueError as exc:
            exc_text = str(exc).lower()
            if (
                "predict_proba" not in exc_text
                and "probability predictions" not in exc_text
            ):
                raise
            if checkpoint_stage == "before_predict" and snapshot.defense is not None:
                snapshot._model = snapshot.apply_defense(data)
            train_pred = snapshot.predict(data.X_train)
            test_pred = snapshot.predict(data.X_test)
            if snapshot.classifier:
                train_scores = snapshot._classification_scores(
                    data.y_train,
                    train_pred,
                )
                test_scores = snapshot._classification_scores(
                    data.y_test,
                    test_pred,
                )
            else:
                train_scores = snapshot._regression_scores(
                    data.y_train,
                    train_pred,
                )
                test_scores = snapshot._regression_scores(
                    data.y_test,
                    test_pred,
                )
            snapshot.score_dict.update(
                {f"training_{key}": value for key, value in train_scores.items()},
            )
            snapshot.score_dict.update(test_scores)
        return dict(snapshot.score_dict or {})

    def _score_epoch_snapshot(self, epoch_index: int, data, attack_config=None):
        if data is None:
            return

        epoch_entry = self.score_dict["epochs"].setdefault(epoch_index, {})
        snapshot = self._build_checkpoint_snapshot()

        benign_start = time.perf_counter()
        benign_scores = self._score_checkpoint_snapshot(snapshot, data)
        benign_time = time.perf_counter() - benign_start

        epoch_entry["benign_scores"] = benign_scores
        epoch_entry.setdefault("timings", {})["benign_score_time"] = benign_time

        if attack_config is None:
            return

        attack_start = time.perf_counter()
        attack_runner = copy.deepcopy(attack_config)
        attack_scores = attack_runner(
            data=data,
            model=snapshot,
            attack_file=None,
            attack_predictions_file=None,
            score_file=None,
        )
        attack_time = time.perf_counter() - attack_start

        epoch_entry["adversarial_scores"] = dict(attack_scores or {})
        epoch_entry["timings"]["adversarial_score_time"] = attack_time

    @staticmethod
    def _checkpoint_file(
        path_dir: Path,
        prefix: str,
        epoch_index: int,
        suffix: str,
    ) -> Path:
        # Standardized format: <prefix>_<epoch><suffix>
        return path_dir / f"{prefix}_{epoch_index}{suffix}"

    def _persist_checkpoint(
        self,
        epoch_index: int,
        data,
        checkpoint_cfg,
        elapsed_time: float,
    ):
        checkpoint_started = time.perf_counter()
        snapshot = self._build_checkpoint_snapshot()
        snapshot.training_time = elapsed_time
        snapshot.training_n = (
            len(data.y_train) if data is not None else self.training_n
        )

        model_path = self._checkpoint_file(
            checkpoint_cfg["dir"],
            checkpoint_cfg["prefix"],
            epoch_index,
            ".pt",
        )
        if model_path.exists():
            model_path.unlink()
        model_save_started = time.perf_counter()
        snapshot.save_model(snapshot._model, str(model_path))
        model_save_time = time.perf_counter() - model_save_started

        epoch_timings = {}
        epochs_payload = self.score_dict.get("epochs", {}) if self.score_dict else {}
        epoch_payload = epochs_payload.get(epoch_index)
        if not isinstance(epoch_payload, dict):
            epoch_payload = epochs_payload.get(str(epoch_index), {})
        if isinstance(epoch_payload, dict):
            raw_timings = epoch_payload.get("timings", {})
            if isinstance(raw_timings, dict):
                epoch_timings = dict(raw_timings)

        record = {
            "epoch": epoch_index,
            # Checkpoints persist model state only; config YAML artifacts are not emitted.
            "model_file": str(model_path),
            "model_state_file": str(model_path),
            "training_elapsed_time": float(elapsed_time),
            "timings": {
                "model_save_time": float(model_save_time),
                **epoch_timings,
            },
        }

        if checkpoint_cfg["score"] and data is not None:
            score_path = self._checkpoint_file(
                checkpoint_cfg["dir"],
                checkpoint_cfg["prefix"],
                epoch_index,
                ".json",
            )
            if score_path.exists():
                score_path.unlink()
            score_started = time.perf_counter()
            checkpoint_scores = self._score_checkpoint_snapshot(snapshot, data)
            score_time = time.perf_counter() - score_started
            checkpoint_scores = {
                **dict(checkpoint_scores or {}),
                "checkpoint_epoch": epoch_index,
                "checkpoint_training_elapsed_time": float(elapsed_time),
                "checkpoint_timings": {
                    "model_save_time": float(model_save_time),
                    "score_time": float(score_time),
                    **epoch_timings,
                },
            }
            snapshot.save_scores(checkpoint_scores, str(score_path))
            record["score_file"] = str(score_path)
            record["timings"]["score_time"] = float(score_time)

        record["timings"]["checkpoint_time"] = float(
            time.perf_counter() - checkpoint_started,
        )

        self.checkpoint_records.append(record)

    def _run_training_epochs(
        self,
        X,
        y,
        *,
        criterion,
        optimizer,
        batch_size,
        epochs,
        batch_losses,
        epoch_offset=0,
    ):
        """Run training epochs and track per-epoch metrics.

        Args:
            epoch_offset: Starting epoch number (for logging and metrics)

        Returns:
            dict mapping epoch numbers to their metrics
        """
        self._model.train()
        epoch_metrics = {}

        from torch.utils.data import DataLoader, Dataset, Subset

        is_tensor_input = isinstance(X, torch.Tensor)
        is_dataloader = (
            hasattr(X, "batch_size") and hasattr(X, "__iter__") and not is_tensor_input
        )
        is_dataset = (
            isinstance(X, (Dataset, Subset))
            and not is_dataloader
            and not is_tensor_input
        )

        for epoch_num in range(epochs):
            epoch_idx = epoch_offset + epoch_num + 1
            epoch_start = time.perf_counter()
            epoch_losses = []

            if is_tensor_input:
                # Tensor input
                for i in range(0, len(X), batch_size):
                    batch_X = X[i : i + batch_size].to(self.device)
                    batch_y = y[i : i + batch_size].to(self.device)
                    optimizer.zero_grad()
                    outputs = self._model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss_val = float(loss.detach().item())
                    batch_losses.append(loss_val)
                    epoch_losses.append(loss_val)
                    loss.backward()
                    optimizer.step()
            else:
                # DataLoader or Dataset/Subset input
                if is_dataset:
                    loader = DataLoader(X, batch_size=batch_size, shuffle=False)
                else:
                    loader = X  # Already a DataLoader
                for batch in loader:
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        batch_X, batch_y = batch[:2]
                    else:
                        raise ValueError("Each batch must be (X, y)")
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = self._model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss_val = float(loss.detach().item())
                    batch_losses.append(loss_val)
                    epoch_losses.append(loss_val)
                    loss.backward()
                    optimizer.step()

            epoch_time = time.perf_counter() - epoch_start
            epoch_loss_mean = float(np.mean(epoch_losses)) if epoch_losses else None

            epoch_metrics[epoch_idx] = {
                "loss": epoch_loss_mean,
                "time": epoch_time,
                "batches": len(epoch_losses),
            }

            logger.info(
                f"Epoch {epoch_idx}/{epoch_offset + epochs}: loss={epoch_loss_mean:.6f}, time={epoch_time:.3f}s",
            )

        return epoch_metrics

    def apply_defense(
        self,
        data: "DataConfig",
        stage: str = "post_fit_pre_predict",
    ) -> EstimatorLike:
        """Override to pre-wrap with a properly configured PyTorchClassifier/Regressor.

        The base-class defense pipeline receives the raw ``torch.nn.Module`` as
        ``estimator``.  For PyTorch models we instead pass a fully-configured
        ART wrapper (built via :meth:`get_art_model`) so that any preprocessor
        or postprocessor defenses attach to the correct ART estimator and
        benefit from the model's criterion, optimizer, and device settings.
        """
        if self.defense is None:
            return self._model

        if self._model is None:
            raise ValueError(
                "PytorchModelConfig must have a fitted model before applying defense",
            )

        # Build the ART wrapper using this model's criterion/optimizer config.
        art_estimator = self.get_art_model(data)

        defense_pipeline = self._require_defense_pipeline()
        if defense_pipeline is None:
            return art_estimator

        stage = defense_pipeline.resolve_stage(
            default_stage=stage,
            model=self,
            data=data,
        )
        defended_estimator = defense_pipeline.apply(
            estimator=art_estimator,
            data=data,
            stage=stage,
        )
        self.defense_application_time = getattr(
            defense_pipeline,
            "defense_application_time",
            None,
        )
        if getattr(defense_pipeline, "score_dict", None):
            if self.score_dict is None:
                self.score_dict = {}
            self.score_dict.update(defense_pipeline.score_dict)
        # Re-apply device overrides so any newly created preprocessing ops are
        # placed on the correct device (particularly important for MPS).
        if isinstance(
            defended_estimator,
            (PyTorchClassifier, PyTorchRegressor),
        ):
            defended_estimator = self._override_art_internal_device(
                defended_estimator,
            )
        return defended_estimator

    def _load_or_train_model(self, data, model_file, times):
        self._validate_torch_data(data)
        self._checkpoint_context = {
            "data": data,
            "model_file": model_file,
            "attack": self._epoch_attack,
        }
        try:
            return super()._load_or_train_model(data, model_file, times)
        finally:
            self._checkpoint_context = None

    def _validate_torch_data(self, data) -> None:
        """Raise TypeError if data contains non-torch tensors/DataLoaders."""
        bad_attrs = []
        from torch.utils.data import Dataset, Subset

        data_loader_types = (TorchDataLoader,) if TorchDataLoader is not None else ()
        for attr in ("X_train", "X_test", "y_train", "y_test"):
            value = getattr(data, attr, None)
            if value is None:
                continue
            if not isinstance(
                value,
                (torch.Tensor, Subset, Dataset, *data_loader_types),
            ):
                bad_attrs.append(f"{attr}: {type(value).__name__}")

        if bad_attrs:
            raise TypeError(
                "PytorchModelConfig requires torch.Tensor or DataLoader inputs, "
                f"but received non-torch types for: {', '.join(bad_attrs)}. "
                "Use PytorchDataConfig (or another torch-compatible data config) "
                "to produce torch tensors before passing data to a torch model.",
            )

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Train the PyTorch model with per-epoch logging and metrics tracking."""
        if self._model is None:
            raise ValueError("Model not initialized")

        start_time = time.perf_counter()
        logger.info(f"Starting training with {len(y)} samples")

        criterion = initialize_criterion(self.criterion)
        optimizer = initialize_optimizer(
            self.optimizer,
            self._model.parameters(),
        )

        nb_epochs = self.fit_params.get("nb_epochs", 1)
        batch_size = self.fit_params.get("batch_size", 32)
        batch_losses = []
        self.checkpoint_records = []

        # Initialize score_dict with epochs subdictionary
        if self.score_dict is None:
            self.score_dict = {}
        if "epochs" not in self.score_dict:
            self.score_dict["epochs"] = {}

        checkpoint_context = self._checkpoint_context or {}
        checkpoint_cfg = self._resolve_checkpoint_config(
            model_file=checkpoint_context.get("model_file", None),
        )
        checkpoint_data = checkpoint_context.get("data", None)
        checkpoint_attack = checkpoint_context.get("attack", None)

        logger.info(
            "Training for %s epochs%s",
            nb_epochs,
            (
                f" with checkpointing every {checkpoint_cfg['every']} epochs"
                if checkpoint_cfg is not None
                else " without checkpointing"
            ),
        )

        for epoch_index in range(1, nb_epochs + 1):
            epoch_metrics = self._run_training_epochs(
                X,
                y,
                criterion=criterion,
                optimizer=optimizer,
                batch_size=batch_size,
                epochs=1,
                batch_losses=batch_losses,
                epoch_offset=epoch_index - 1,
            )
            self.score_dict["epochs"].update(epoch_metrics)
            self.training_n = len(y)

            if checkpoint_data is not None:
                self._score_epoch_snapshot(
                    epoch_index=epoch_index,
                    data=checkpoint_data,
                    attack_config=checkpoint_attack,
                )

            if checkpoint_cfg is not None:
                should_checkpoint = (epoch_index % checkpoint_cfg["every"]) == 0
                is_final = epoch_index == nb_epochs
                if is_final and checkpoint_cfg["include_final"]:
                    should_checkpoint = True
                if should_checkpoint:
                    elapsed = time.perf_counter() - start_time
                    logger.info("Checkpointing at epoch %s", epoch_index)
                    self._persist_checkpoint(
                        epoch_index,
                        checkpoint_data,
                        checkpoint_cfg,
                        elapsed,
                    )

        end_time = time.perf_counter()
        self.training_time = end_time - start_time
        self.training_n = len(y)

        # Compute final loss from all batches
        final_loss = float(np.mean(batch_losses)) if batch_losses else None
        self.score_dict["optimizer_loss"] = final_loss
        self.score_dict["training_time"] = self.training_time

        if len(self.checkpoint_records) > 0:
            self.score_dict["checkpoints"] = copy.deepcopy(
                self.checkpoint_records,
            )

        logger.info(
            "Training completed: loss=%0.6f, time=%0.2fs, epochs=%s",
            final_loss if final_loss is not None else float("nan"),
            self.training_time,
            nb_epochs,
        )

    def predict(
        self,
        X: Union[torch.Tensor, torch.utils.data.DataLoader],
    ) -> torch.Tensor:
        """Make predictions, handling Tensor, DataLoader, Subset, or Dataset inputs."""
        if self._model is None:
            raise ValueError("Model not initialized")

        from torch.utils.data import DataLoader, Dataset, Subset

        is_tensor_input = isinstance(X, torch.Tensor)
        is_dataloader = (
            hasattr(X, "batch_size") and hasattr(X, "__iter__") and not is_tensor_input
        )
        is_dataset = (
            isinstance(X, (Dataset, Subset))
            and not is_dataloader
            and not is_tensor_input
        )

        # ART wrappers (e.g., PyTorchClassifier) expose predict() instead of eval().
        if hasattr(self._model, "predict") and not hasattr(self._model, "eval"):
            if is_dataloader or is_dataset:
                loader = (
                    X
                    if is_dataloader
                    else DataLoader(X, batch_size=128, shuffle=False)
                )
                x_batches = []
                for batch in loader:
                    batch_x = batch[0] if isinstance(batch, (tuple, list)) else batch
                    x_batches.append(batch_x)
                if len(x_batches) == 0:
                    return torch.empty(0)
                x_tensor = torch.cat(x_batches, dim=0)
            else:
                x_tensor = X
            x_np = (
                x_tensor.detach().cpu().numpy()
                if isinstance(x_tensor, torch.Tensor)
                else np.asarray(x_tensor)
            )
            if np.issubdtype(x_np.dtype, np.floating):
                x_np = x_np.astype(ART_NUMPY_DTYPE, copy=False)
            y_pred = self._model.predict(x_np)
            return torch.as_tensor(y_pred)

        self._model.eval()
        predictions = []

        with torch.no_grad():
            if is_dataloader or is_dataset:
                loader = (
                    X
                    if is_dataloader
                    else DataLoader(X, batch_size=128, shuffle=False)
                )
                for batch in loader:
                    if isinstance(batch, (tuple, list)):
                        batch_X = batch[0]
                    else:
                        batch_X = batch
                    batch_X = batch_X.to(self.device)
                    batch_pred = self._model(batch_X)
                    predictions.append(batch_pred.cpu())
            else:
                X = X.to(self.device)
                predictions.append(self._model(X).cpu())

        y_pred = torch.cat(predictions, dim=0)
        return y_pred

    def _classification_scores(self, y_true, y_pred) -> dict:
        """Compute classification scores from predictions."""
        y_true_np = (
            y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        )

        if isinstance(y_pred, torch.Tensor):
            if y_pred.ndim > 1:
                y_pred_np = y_pred.argmax(dim=1).cpu().numpy()
            else:
                y_pred_np = y_pred.cpu().numpy()
        else:
            y_pred_np = y_pred

        scores = {
            "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
            "precision": float(
                precision_score(
                    y_true_np,
                    y_pred_np,
                    average="weighted",
                    zero_division=0,
                ),
            ),
            "recall": float(
                recall_score(
                    y_true_np,
                    y_pred_np,
                    average="weighted",
                    zero_division=0,
                ),
            ),
            "f1": float(
                f1_score(
                    y_true_np,
                    y_pred_np,
                    average="weighted",
                    zero_division=0,
                ),
            ),
        }
        return scores

    def _regression_scores(self, y_true, y_pred) -> dict:
        """Compute regression scores from predictions."""
        y_true_np = (
            y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        )
        y_pred_np = (
            y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
        )

        mse = float(np.mean((y_true_np - y_pred_np) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true_np - y_pred_np)))

        scores = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }
        return scores

    def get_art_model(
        self,
        data: "DataConfig",
    ) -> PyTorchClassifier | PyTorchRegressor:
        """Get ART-compatible model wrapper for adversarial robustness.

        Args:
            data: Runtime data configuration used for shape and loader inference.

        Returns:
            ART classifier or regressor wrapper around the configured torch model.
        """
        if self.clip_values is None or len(self.clip_values) == 0:
            clip_values = (0.0, 1.0)
        else:
            clip_values = self.clip_values

        from torch.utils.data import DataLoader, Dataset, Subset

        # Use data.batch_size if available, else fallback to 32
        batch_size = getattr(data, "batch_size", None) or self.fit_params.get(
            "batch_size",
            32,
        )
        # Always use a DataLoader for shape inference and ART
        if isinstance(data.X_train, torch.utils.data.DataLoader):
            loader = data.X_train
        elif isinstance(data.X_train, (Dataset, Subset)):
            loader = DataLoader(data.X_train, batch_size=batch_size, shuffle=False)
        else:
            loader = None

        if loader is not None:
            batch = next(iter(loader))
            if isinstance(batch, (tuple, list)):
                input_shape = batch[0].shape[1:]
            else:
                input_shape = batch.shape[1:]
        else:
            input_shape = data.X_train.shape[1:]

        nb_classes = len(torch.unique(data.y_train))
        art_model = self._model_for_art()
        if isinstance(art_model, (PyTorchClassifier, PyTorchRegressor)):
            return self._override_art_internal_device(art_model)
        art_device_type = self._resolve_art_device_type()
        logger.info(f"[ART] Using batch_size={batch_size} for ART estimator.")
        if self.classifier:
            estimator = PyTorchClassifier(
                model=art_model,
                loss=initialize_criterion(self.criterion),
                optimizer=initialize_optimizer(
                    self.optimizer,
                    art_model.parameters(),
                ),
                input_shape=input_shape,
                nb_classes=nb_classes,
                clip_values=clip_values,
                preprocessing=None,
                device_type=art_device_type,
            )
            return self._override_art_internal_device(estimator)
        else:
            estimator = PyTorchRegressor(
                model=art_model,
                loss=initialize_criterion(self.criterion),
                optimizer=initialize_optimizer(
                    self.optimizer,
                    art_model.parameters(),
                ),
                input_shape=input_shape,
                nb_classes=nb_classes,
                clip_values=clip_values,
                preprocessing=None,
                device_type=art_device_type,
            )
            return self._override_art_internal_device(estimator)
