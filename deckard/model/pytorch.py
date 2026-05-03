# OS imports
import copy
import logging
import time
from pathlib import Path

# Typing imports
from dataclasses import dataclass, field
from omegaconf import DictConfig
from typing import Any, Union
import numpy as np

# Torch imports
import torch

# Sklearn imports
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ART imports
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor

from . import ModelConfig
from ..data import DataConfig
from ..utils import load_class, resolve_torch_device

logger = logging.getLogger(__name__)

ScorerDictConfig = Any

__all__ = ["PytorchModelConfig"]


def initialize_criterion(criterion_spec):
    """Initialize criterion from string or dict using load_class."""
    if isinstance(criterion_spec, str):
        if "." not in criterion_spec and ":" not in criterion_spec:
            criterion_spec = f"torch.nn.{criterion_spec}"
        return load_class(criterion_spec)
    elif isinstance(criterion_spec, (dict, DictConfig)):
        criterion_name = criterion_spec.get("name") or criterion_spec.get(
            "_target_"
        )
        criterion_params = {
            k: v
            for k, v in criterion_spec.items()
            if k not in ["name", "_target_"]
        }
        return load_class(criterion_name, **criterion_params)
    else:
        raise ValueError(
            f"criterion must be str or dict, got {type(criterion_spec)}"
        )


def initialize_optimizer(optimizer_spec, model_params):
    """Initialize optimizer from string or dict using load_class."""
    if isinstance(optimizer_spec, str):
        if "." not in optimizer_spec and ":" not in optimizer_spec:
            optimizer_spec = f"torch.optim.{optimizer_spec}"
        return load_class(optimizer_spec, model_params)
    elif isinstance(optimizer_spec, (dict, DictConfig)):
        optimizer_name = optimizer_spec.get("name") or optimizer_spec.get(
            "_target_"
        )
        if (
            isinstance(optimizer_name, str)
            and "." not in optimizer_name
            and ":" not in optimizer_name
        ):
            optimizer_name = f"torch.optim.{optimizer_name}"
        optimizer_params = {
            k: v
            for k, v in optimizer_spec.items()
            if k not in ["name", "_target_"]
        }
        optimizer_params["params"] = model_params
        return load_class(optimizer_name, **optimizer_params)
    else:
        raise ValueError(
            f"optimizer must be str or dict, got {type(optimizer_spec)}"
        )


@dataclass(eq=False)
class PytorchModelConfig(ModelConfig):
    """Configuration for PyTorch models using load_class for generic instantiation.

    Attributes:
        model_type: Fully qualified class name (e.g., "torch.nn.Linear" or "torchvision.models.resnet18")
        model_params: Constructor parameters for the model
        device: torch.device ("cpu", "cuda", etc.)
        criterion: Loss function spec (str name or dict with _target_)
        optimizer: Optimizer spec (str name or dict with _target_)
        fit_params: Parameters for model training
        classifier: Whether model is classifier (True) or regressor (False)
    """

    model_type: str = "torch.nn.Linear"
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

    def __post_init__(self):
        """Initialize model using load_class, following parent pattern."""
        if isinstance(self.scorer, str) and self.scorer.lower() in {
            "auto",
            "default",
        }:
            scorer_cls = (
                "deckard.score.base.DefaultPytorchClassifierConfig"
                if self.classifier
                else "deckard.score.base.DefaultPytorchRegressorConfig"
            )
            self.scorer = load_class(scorer_cls)

        self.device = self._resolve_torch_device(self.device)

        torch.manual_seed(self.random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_seed)

        # Call parent __post_init__ for shared initialization
        super().__post_init__()

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
            art_estimator._model, "to"
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

                art_estimator._apply_preprocessing = (
                    _dtype_safe_apply_preprocessing
                )
                art_estimator._deckard_dtype_wrapped = True

        return art_estimator

    def set_epoch_attack(self, attack_config: Any) -> None:
        self._epoch_attack = attack_config

    def _initialize_model(self):
        """Initialize PyTorch model using load_class."""
        if self.model_params is not None:
            self._model = load_class(self.model_type, **self.model_params)
        else:
            self._model = load_class(self.model_type)

        # Move model to device
        self._model = self._model.to(self.device)
        logger.info(
            f"Initialized model {self.model_type} on device {self.device}"
        )

    def get_model(self):
        """Return the underlying PyTorch model."""
        if self._model is None:
            raise ValueError("Model not initialized")
        return self._model

    def save(self, filepath: str) -> None:
        """Serialize PyTorch model state and config metadata."""
        if self._model is None:
            raise ValueError("Model not initialized")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(
                f"File {filepath} already exists. Will not overwrite."
            )

        payload = {
            "model_type": self.model_type,
            "model_params": self.model_params,
            "classifier": self.classifier,
            "fit_params": self.fit_params,
            "criterion": self.criterion,
            "optimizer": self.optimizer,
            "clip_values": self.clip_values,
            "random_seed": self.random_seed,
            "channels_first": self.channels_first,
            "library": self.library,
            "alias": self.alias,
            "device": str(self.device),
            "score_dict": self.score_dict,
            "checkpoint_records": self.checkpoint_records,
            "training_time": self.training_time,
            "prediction_time": self.prediction_time,
            "training_n": self.training_n,
            "prediction_n": self.prediction_n,
            "state_dict": self._model.state_dict(),
        }
        torch.save(payload, path)

    def load(self, filepath: str) -> "PytorchModelConfig":
        """Load PyTorch model state and config metadata."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(filepath)

        payload = torch.load(path, map_location=self.device)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise TypeError(f"Unsupported serialized payload in {filepath}")

        self.model_type = payload.get("model_type", self.model_type)
        self.model_params = payload.get("model_params", self.model_params)
        self.classifier = payload.get("classifier", self.classifier)
        self.fit_params = payload.get("fit_params", self.fit_params)
        self.criterion = payload.get("criterion", self.criterion)
        self.optimizer = payload.get("optimizer", self.optimizer)
        self.clip_values = payload.get("clip_values", self.clip_values)
        self.random_seed = payload.get("random_seed", self.random_seed)
        self.channels_first = payload.get("channels_first", self.channels_first)
        self.library = payload.get("library", self.library)
        self.alias = payload.get("alias", self.alias)
        self.score_dict = payload.get("score_dict", self.score_dict)
        self.checkpoint_records = payload.get(
            "checkpoint_records",
            self.checkpoint_records,
        )
        self.training_time = payload.get("training_time", self.training_time)
        self.prediction_time = payload.get(
            "prediction_time", self.prediction_time
        )
        self.training_n = payload.get("training_n", self.training_n)
        self.prediction_n = payload.get("prediction_n", self.prediction_n)

        self._initialize_model()
        self._model.load_state_dict(payload["state_dict"])
        self._model = self._model.to(self.device)
        return self

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
            checkpoint_dir = (
                model_path.parent / f"{model_path.stem}_checkpoints"
            )
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_prefix = self.fit_params.get("checkpoint_prefix", None)
        if checkpoint_prefix in {None, ""}:
            if model_file is not None:
                checkpoint_prefix = Path(model_file).stem
            elif self.alias not in {None, ""}:
                checkpoint_prefix = self.alias
            else:
                checkpoint_prefix = (
                    str(self.model_type).split(":")[-1].split(".")[-1]
                )

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
        snapshot._model.load_state_dict(copy.deepcopy(self._model.state_dict()))
        snapshot._model = snapshot._model.to(snapshot.device)
        return snapshot

    def _score_checkpoint_snapshot(self, snapshot, data):
        # Copy over epoch metrics from our score_dict if they exist
        if "epochs" in self.score_dict:
            snapshot.score_dict["epochs"] = copy.deepcopy(
                self.score_dict["epochs"]
            )
        checkpoint_stage = None
        if snapshot.defense is not None:
            defense_pipeline = snapshot._require_defense_pipeline()
            checkpoint_stage = defense_pipeline.resolve_stage(
                default_stage="post_fit_pre_predict",
                model=snapshot,
                data=data,
            )
            if checkpoint_stage == "post_fit_pre_predict":
                snapshot._model = snapshot._apply_defense(data)
        try:
            snapshot._evaluate_and_score(data, times={})
        except ValueError as exc:
            if "predict_proba" not in str(exc):
                raise
            if (
                checkpoint_stage == "before_predict"
                and snapshot.defense is not None
            ):
                snapshot._model = snapshot._apply_defense(data)
            train_pred = snapshot._predict(data.X_train)
            test_pred = snapshot._predict(data.X_test)
            if snapshot.classifier:
                train_scores = snapshot._classification_scores(
                    data.y_train, train_pred
                )
                test_scores = snapshot._classification_scores(
                    data.y_test, test_pred
                )
            else:
                train_scores = snapshot._regression_scores(
                    data.y_train, train_pred
                )
                test_scores = snapshot._regression_scores(
                    data.y_test, test_pred
                )
            snapshot.score_dict.update(
                {
                    f"training_{key}": value
                    for key, value in train_scores.items()
                },
            )
            snapshot.score_dict.update(test_scores)
        return dict(snapshot.score_dict or {})

    def _score_epoch_snapshot(self, epoch_index: int, data, attack_config=None):
        if data is None:
            return

        epoch_entry = self.score_dict["epochs"].setdefault(epoch_index, {})
        snapshot = self._build_checkpoint_snapshot()

        benign_start = time.process_time()
        benign_scores = self._score_checkpoint_snapshot(snapshot, data)
        benign_time = time.process_time() - benign_start

        epoch_entry["benign_scores"] = benign_scores
        epoch_entry.setdefault("timings", {})["benign_score_time"] = benign_time

        if attack_config is None:
            return

        attack_start = time.process_time()
        attack_runner = copy.deepcopy(attack_config)
        attack_scores = attack_runner(
            data=data,
            model=snapshot,
            attack_file=None,
            attack_predictions_file=None,
            score_file=None,
        )
        attack_time = time.process_time() - attack_start

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
        snapshot = self._build_checkpoint_snapshot()
        snapshot.training_time = elapsed_time
        snapshot.training_n = (
            len(data.y_train) if data is not None else self.training_n
        )

        model_path = self._checkpoint_file(
            checkpoint_cfg["dir"],
            checkpoint_cfg["prefix"],
            epoch_index,
            ".pkl",
        )
        if model_path.exists():
            model_path.unlink()
        snapshot.save(str(model_path))

        record = {
            "epoch": epoch_index,
            "model_file": str(model_path),
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
            checkpoint_scores = self._score_checkpoint_snapshot(snapshot, data)
            snapshot.save_scores(checkpoint_scores, str(score_path))
            record["score_file"] = str(score_path)

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

        for epoch_num in range(epochs):
            epoch_idx = epoch_offset + epoch_num + 1
            epoch_start = time.process_time()
            epoch_losses = []

            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size].to(self.device)  # noqa E203
                batch_y = y[i : i + batch_size].to(self.device)  # noqa E203

                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss_val = float(loss.detach().item())
                batch_losses.append(loss_val)
                epoch_losses.append(loss_val)
                loss.backward()
                optimizer.step()

            epoch_time = time.process_time() - epoch_start
            epoch_loss_mean = (
                float(np.mean(epoch_losses)) if epoch_losses else None
            )

            epoch_metrics[epoch_idx] = {
                "loss": epoch_loss_mean,
                "time": epoch_time,
                "batches": len(epoch_losses),
            }

            logger.info(
                f"Epoch {epoch_idx}/{epoch_offset + epochs}: loss={epoch_loss_mean:.6f}, time={epoch_time:.3f}s",
            )

        return epoch_metrics

    def _apply_defense(self, data):
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
            default_stage="post_fit_pre_predict",
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
            defended_estimator, (PyTorchClassifier, PyTorchRegressor)
        ):
            defended_estimator = self._override_art_internal_device(
                defended_estimator
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
        from torch.utils.data import DataLoader as _DL

        bad_attrs = []
        for attr in ("X_train", "X_test", "y_train", "y_test"):
            value = getattr(data, attr, None)
            if value is None:
                continue
            if not isinstance(value, (torch.Tensor, _DL)):
                bad_attrs.append(f"{attr}: {type(value).__name__}")

        if bad_attrs:
            raise TypeError(
                "PytorchModelConfig requires torch.Tensor or DataLoader inputs, "
                f"but received non-torch types for: {', '.join(bad_attrs)}. "
                "Use PytorchDataConfig (or another torch-compatible data config) "
                "to produce torch tensors before passing data to a torch model.",
            )

    def _train(self, X: torch.Tensor, y: torch.Tensor):
        """Train the PyTorch model with per-epoch logging and metrics tracking."""
        if self._model is None:
            raise ValueError("Model not initialized")

        start_time = time.process_time()
        logger.info(f"Starting training with {len(y)} samples")

        criterion = initialize_criterion(self.criterion)
        optimizer = initialize_optimizer(
            self.optimizer, self._model.parameters()
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
                    elapsed = time.process_time() - start_time
                    logger.info("Checkpointing at epoch %s", epoch_index)
                    self._persist_checkpoint(
                        epoch_index,
                        checkpoint_data,
                        checkpoint_cfg,
                        elapsed,
                    )

        end_time = time.process_time()
        self.training_time = end_time - start_time
        self.training_n = len(y)

        # Compute final loss from all batches
        final_loss = float(np.mean(batch_losses)) if batch_losses else None
        self.score_dict["optimizer_loss"] = final_loss
        self.score_dict["training_time"] = self.training_time

        if len(self.checkpoint_records) > 0:
            self.score_dict["checkpoints"] = copy.deepcopy(
                self.checkpoint_records
            )

        logger.info(
            "Training completed: loss=%0.6f, time=%0.2fs, epochs=%s",
            final_loss if final_loss is not None else float("nan"),
            self.training_time,
            nb_epochs,
        )

    def _predict(self, X: Union[torch.Tensor, torch.utils.data.DataLoader]):
        """Make predictions, handling both Tensor and DataLoader inputs."""
        if self._model is None:
            raise ValueError("Model not initialized")

        # ART wrappers (e.g., PyTorchClassifier) expose predict() instead of eval().
        if hasattr(self._model, "predict") and not hasattr(self._model, "eval"):
            if isinstance(X, torch.utils.data.DataLoader):
                x_batches = []
                for batch in X:
                    batch_x = (
                        batch[0] if isinstance(batch, (tuple, list)) else batch
                    )
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
            if isinstance(X, torch.utils.data.DataLoader):
                for batch in X:
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
                    y_true_np, y_pred_np, average="weighted", zero_division=0
                ),
            ),
            "f1": float(
                f1_score(
                    y_true_np, y_pred_np, average="weighted", zero_division=0
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

    def get_art_model(self, data: DataConfig):
        """Get ART-compatible model wrapper for adversarial robustness."""
        if self.clip_values is None or len(self.clip_values) == 0:
            clip_values = (0.0, 1.0)
        else:
            clip_values = self.clip_values

        if isinstance(data.X_train, torch.utils.data.DataLoader):
            batch = next(iter(data.X_train))
            if isinstance(batch, (tuple, list)):
                input_shape = batch[0][0].shape[1:]
            else:
                input_shape = batch[0].shape[1:]
        else:
            input_shape = data.X_train.shape[1:]

        nb_classes = len(torch.unique(data.y_train))
        art_model = self._model_for_art()
        if isinstance(art_model, (PyTorchClassifier, PyTorchRegressor)):
            return self._override_art_internal_device(art_model)
        art_device_type = self._resolve_art_device_type()
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
