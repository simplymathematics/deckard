import time
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pandas as pd
from art.config import ART_NUMPY_DTYPE
from sklearn.base import BaseEstimator

from ...data._mixins import RuntimePayload, SensitiveColumnsMixin
from ...model.base import ModelConfig, logger
from ...model.defense.base import DefenseConfig
from ...pytorch.model import PytorchModelConfig
from ...utils import (
    is_default_config_value,
    load_class,
    resolve_class,
)
from .data import FairlearnDataConfig

try:
    import torch as torch_module
    import torch.nn as nn_module
except ImportError:
    torch_module = None
    nn_module = None


class FairnessBehaviorMixin:
    """Shared fairness-aware model behavior used by sklearn and PyTorch configs.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def apply_fairness_behavior(self) -> None:
        """Public entrypoint for fairlearn-specific model behavior setup."""
        self.__post_init__()

    def __post_init__(self):
        if (
            is_default_config_value(getattr(self, "scorer", None), include_best=False)
            or getattr(self, "scorer", None) is None
        ):
            from .score import (
                DefaultFairlearnClassificationScorerDictConfig,
                DefaultFairlearnRegressionScorerDictConfig,
            )

            if hasattr(self, "classifier") and self.classifier is False:
                self.scorer = DefaultFairlearnRegressionScorerDictConfig()
            else:
                self.scorer = DefaultFairlearnClassificationScorerDictConfig()

        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the wrapped Fairlearn model, forwarding sensitive features when supported.

        Args:
            X: Training feature matrix.
            y: Training target vector.

        Raises:
            ValueError: If the wrapped model has not been initialized.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        fit_method = getattr(self._model, "fit", None)
        if not callable(fit_method):
            return super().train(X, y)

        start_time = time.perf_counter()
        fit_params = getattr(self, "fit_params", None) or {}
        sensitive = self._resolve_sensitive_features_for_batch(y, split="train")
        if (
            sensitive is not None
            and self._method_accepts_sensitive_features(fit_method)
            and "sensitive_features" not in fit_params
        ):
            fit_params = {**fit_params, "sensitive_features": sensitive}
        fit_method(X, y, **fit_params)
        self.training_time = time.perf_counter() - start_time
        self.training_n = len(y)
        logger.info(f"Model trained in {self.training_time:.2f} seconds")

    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions from the wrapped Fairlearn model.

        Args:
            X: Feature matrix for inference.

        Returns:
            Prediction payload returned by the wrapped model.

        Raises:
            ValueError: If the wrapped model has not been initialized.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        predict_method = getattr(self._model, "predict", None)
        if not callable(predict_method):
            return super().predict(X)
        sensitive = self._resolve_sensitive_features_for_batch(X, split="test")
        try:
            return self._call_with_optional_sensitive(
                predict_method,
                X,
                sensitive,
            )
        except TypeError as exc:
            if "loop of ufunc does not support argument" in str(
                exc,
            ) or "can't convert" in str(exc):
                X_array = np.array(X, dtype=ART_NUMPY_DTYPE)
                return self._call_with_optional_sensitive(
                    self._model.predict,
                    X_array,
                    sensitive,
                )
            raise

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Generate class probabilities from the wrapped Fairlearn model.

        Args:
            X: Feature matrix for inference.

        Returns:
            Probability payload returned by the wrapped model.

        Raises:
            ValueError: If the wrapped model has not been initialized.
        """
        if self._model is None:
            raise ValueError("Model not initialized")
        predict_proba = getattr(self._model, "predict_proba", None)
        if not callable(predict_proba):
            return super().predict_proba(X)
        sensitive = self._resolve_sensitive_features_for_batch(X, split="test")
        return self._call_with_optional_sensitive(
            predict_proba,
            X,
            sensitive,
        )


@dataclass(eq=False, kw_only=True)
class FairlearnModelConfig(
    SensitiveColumnsMixin,
    FairnessBehaviorMixin,
    ModelConfig,
):
    """Fairness-aware model config for sklearn models.

    Inherits sklearn training/prediction from ModelConfig and adds
    fairness-aware scoring and defense support via FairnessBehaviorMixin.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    data: Union[FairlearnDataConfig, None] = None
    fit_params: dict = field(default_factory=dict)


@dataclass(eq=False, kw_only=True)
class FairlearnPytorchModelConfig(
    SensitiveColumnsMixin,
    FairnessBehaviorMixin,
    PytorchModelConfig,
):
    """Fairness-aware model config for PyTorch models.

    Inherits all torch training/prediction/defense from PytorchModelConfig
    and adds fairness-aware scoring via FairnessBehaviorMixin.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    data: Union[FairlearnDataConfig, None] = None


class BinaryLogitAdapter:
    """Adapter that normalizes binary outputs to a single-logit tensor.

    Attributes:
        base_model: Wrapped callable model used for forward prediction.
        _nn_module: Torch module namespace used for optional dynamic subclassing.
    """

    def __init__(self, base_model, nn_module):
        # Dynamically inherit from nn.Module
        self.base_model = base_model
        self._nn_module = nn_module
        if hasattr(nn_module, "Module"):
            self.__class__ = type(
                "_BinaryLogitAdapter",
                (nn_module.Module,),
                dict(self.__class__.__dict__),
            )
            nn_module.Module.__init__(self)

    def forward(self, x):
        """Normalize binary model outputs to a single-logit column.

        Args:
            x: Input tensor batch.

        Returns:
            Two-dimensional single-logit tensor.

        Raises:
            ValueError: If predictor output shape is unsupported.
        """
        out = self.base_model(x)
        if hasattr(out, "ndim"):
            if out.ndim == 1:
                return out.reshape(-1, 1)
            if out.ndim == 2 and out.shape[1] == 1:
                return out
            if out.ndim == 2 and out.shape[1] >= 2:
                return out[:, 1:2]
        raise ValueError(
            f"Unsupported predictor output shape for fairness: {getattr(out, 'shape', None)}",
        )


@dataclass(eq=False, kw_only=True)
class FairlearnDefenseConfig(SensitiveColumnsMixin, DefenseConfig):
    """Fairness-aware defense config that inherits DefenseConfig.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    data: Union[FairlearnDataConfig, None] = None

    def _resolve_fairness_defense_spec(self):
        if hasattr(self, "name"):
            candidate = getattr(self, "name", None)
            if isinstance(candidate, str) and candidate.startswith("fairlearn."):
                return candidate, dict(getattr(self, "defense_params", {}) or {})
        normalized_name = getattr(self, "defense_name", None)
        if isinstance(normalized_name, str) and normalized_name.startswith(
            "fairlearn.",
        ):
            return normalized_name, dict(getattr(self, "defense_params", {}) or {})
        defense_obj = getattr(self, "defense", None)
        if defense_obj is not None:
            nested_name = getattr(defense_obj, "name", None)
            if isinstance(nested_name, str) and nested_name.startswith("fairlearn."):
                return nested_name, dict(
                    getattr(defense_obj, "defense_params", {}) or {},
                )
        return None, {}

    def _apply_fairlearn_defense(self, data):
        defense_name, defense_params = self._resolve_fairness_defense_spec()
        if not defense_name or not defense_name.startswith("fairlearn."):
            raise ValueError(
                "Fairlearn defense helper requires a fairlearn defense name",
            )

        if getattr(self, "data", None) is None:
            self.data = data

        if self._model is None:
            raise ValueError(
                "Fairness model must have a fitted estimator before applying defense",
            )

        module_name, class_name = defense_name.rsplit(".", 1)
        try:
            defense_class = resolve_class(defense_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Could not import defense class {class_name} from {module_name}",
            ) from exc

        constraints = None
        if "constraints" in defense_params:
            constraints_raw = defense_params.pop("constraints")
            if isinstance(constraints_raw, str) and "." in constraints_raw:
                constraints = load_class(constraints_raw)
            elif isinstance(constraints_raw, dict):
                constraints_cfg = dict(constraints_raw)
                c_target = constraints_cfg.pop("_target_", None)
                if c_target is None:
                    raise ValueError("constraints dict must include '_target_'")
                constraints = load_class(c_target, **constraints_cfg)
            else:
                constraints = constraints_raw

        fairlearn_submodule = module_name.split(".")[1]
        base_estimator = self.get_model()

        start = time.perf_counter()
        if fairlearn_submodule == "reductions":
            if constraints is None:
                raise ValueError(
                    "fairlearn.reductions defenses require a 'constraints' key in defense parameters",
                )
            defended_estimator = defense_class(
                base_estimator,
                constraints,
                **defense_params,
            )
        elif fairlearn_submodule == "postprocessing":
            if constraints is not None and "constraints" not in defense_params:
                defense_params["constraints"] = constraints
            defended_estimator = defense_class(
                estimator=base_estimator,
                **defense_params,
            )
        elif fairlearn_submodule == "adversarial":
            if constraints is not None and "constraints" not in defense_params:
                defense_params["constraints"] = constraints
            defended_estimator = defense_class(**defense_params)
        else:
            raise NotImplementedError(
                f"Fairlearn submodule '{fairlearn_submodule}' is not supported. "
                "Expected one of: reductions, postprocessing, adversarial.",
            )

        self._apply_fit = True
        if self._apply_fit:
            defended_estimator = self._fit_defended_estimator(
                defended_estimator,
                data,
            )
        self.defense_application_time = time.perf_counter() - start
        return defended_estimator

    def train(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        if self._model is None:
            model_name = str(self.resolve_name(default="") or "").strip()
            if model_name == "":
                raise ValueError("FairlearnDefenseConfig.name must be set")
            self._model = load_class(
                model_name,
                **(self.model_params if self.model_params is not None else {}),
            )

        fit_params = getattr(self, "fit_params", None) or {}
        sensitive = self._resolve_sensitive_features_for_batch(y, split="train")
        if (
            sensitive is not None
            and self._method_accepts_sensitive_features(self._model.fit)
            and "sensitive_features" not in fit_params
        ):
            fit_params = {**fit_params, "sensitive_features": sensitive}

        self._model.fit(X, y, **fit_params)
        return self._model

    def apply_defense(self, data: RuntimePayload) -> "BaseEstimator":
        """Apply configured fairness defense and return the defended estimator.

        Args:
            data: Runtime data payload consumed by the selected defense.

        Returns:
            Defended estimator instance.
        """
        defense_name, _ = self._resolve_fairness_defense_spec()
        if not defense_name or not defense_name.startswith("fairlearn."):
            return super().apply_defense(data)
        return self._apply_fairlearn_defense(data)

    @staticmethod
    def _adapt_binary_torch_predictor(predictor_model, data):
        """
        Fairlearn binary classification expects a single-score predictor output.
        This adapts a PyTorch model to output a single logit if needed.
        """
        if torch_module is None or nn_module is None:
            return predictor_model

        if not hasattr(predictor_model, "forward"):
            return predictor_model

        y_train = getattr(data, "y_train", None)
        if y_train is None:
            return predictor_model

        if isinstance(y_train, torch_module.Tensor):
            y_values = y_train
            if y_values.unique().numel() != 2:
                return predictor_model
        else:
            # fallback for non-tensor
            import numpy as np

            y_values = np.asarray(y_train)
            if np.unique(y_values).size != 2:
                return predictor_model

        if not FairlearnDefenseConfig._needs_wrap_predictor(
            predictor_model,
            data,
            torch_module,
        ):
            return predictor_model

        return BinaryLogitAdapter(predictor_model, nn_module)

    @staticmethod
    def _needs_wrap_predictor(model, data, torch_module):
        num_classes = getattr(model, "num_classes", None)
        if num_classes == 2:
            return True
        x_train = getattr(data, "X_train", None)
        if (
            x_train is None
            or not isinstance(x_train, torch_module.Tensor)
            or len(x_train) == 0
        ):
            return False
        try:
            with torch_module.no_grad():
                sample = x_train[:1]
                device = next(model.parameters()).device
                out = model(sample.to(device))
            return bool(getattr(out, "ndim", 0) == 2 and out.shape[1] == 2)
        except Exception:
            return False


__all__ = [
    "SensitiveColumnsMixin",
    "FairnessBehaviorMixin",
    "FairlearnModelConfig",
    "FairlearnPytorchModelConfig",
    "FairlearnDefenseConfig",
]
