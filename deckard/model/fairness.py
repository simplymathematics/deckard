from ..utils import is_default_config_value
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from art.config import ART_NUMPY_DTYPE
from sklearn.base import BaseEstimator

from .base import ModelConfig, logger
from .defend import DefenseConfig
from .pytorch import PytorchModelConfig
from ..data.fairness import FairlearnDataConfig
from ..utils import (
    ConfigBase,
    load_class,
    probabilities_from_model_outputs,
    resolve_class,
)
from ..score import ScorerDictConfig

try:
    import torch as torch_module
    import torch.nn as nn_module
except ImportError:
    torch_module = None
    nn_module = None


class _SensitiveBehaviorMixin:
    """Model/defense mixin for explicit fairness-aware training and scoring."""

    data: Any = None
    _model: Any = None
    scorer: Union[ScorerDictConfig, str, None] = None
    classifier: Any = False
    probability: bool = False
    fit_params: Any = None

    def _resolve_runtime_sensitive_source(self, split: str):
        if split == "train":
            return getattr(self.data, "_sensitive_train", None)
        if split == "test":
            return getattr(self.data, "_sensitive_test", None)
        if split == "all":
            return getattr(self.data, "_sensitive_all", None)
        if split == "val":
            raise NotImplementedError(
                "Validation sensitive features are not implemented yet",
            )
        raise ValueError(f"Unsupported fairness split: {split}")

    def _resolve_scoring_split(self, mode: str) -> str:
        if mode == "train":
            return "train"
        if mode in {"test", "attack"}:
            return "test"
        if mode in {"val", "attack-val"}:
            return "val"
        if mode == "all":
            return "all"
        raise ValueError(f"Unsupported fairness scoring mode: {mode}")

    def _validate_sensitive_series(self, sensitive, context: str):
        if sensitive is None:
            return None
        sensitive_series = pd.Series(sensitive)
        if len(sensitive_series) == 0:
            raise ValueError(f"Sensitive features are empty during {context}")
        if sensitive_series.dropna().empty:
            raise ValueError(
                f"Sensitive features are all null during {context}",
            )
        if sensitive_series.astype(str).str.strip().eq("").all():
            raise ValueError(f"Sensitive features are blank during {context}")
        return sensitive_series

    def _infer_split_from_batch(
        self,
        batch,
        scoring_mode: Optional[str] = None,
    ):
        valid_splits = {"train", "test", "val", "all"}
        if scoring_mode is None:
            raise ValueError(
                "scoring_mode must be explicitly provided (one of 'train', 'test', 'val', 'all')",
            )
        if scoring_mode not in valid_splits:
            raise ValueError(
                f"Invalid scoring_mode '{scoring_mode}'. Must be one of {valid_splits}.",
            )
        return scoring_mode

    def _resolve_sensitive_features_for_batch(
        self,
        batch,
        split: Optional[str] = None,
        scoring_mode: Optional[str] = None,
    ):
        if getattr(self, "data", None) is None:
            return None

        n_rows = len(batch)
        batch_index = getattr(batch, "index", None)
        resolved_split = scoring_mode or split or self._infer_split_from_batch(batch)
        if resolved_split is None:
            return None

        sensitive = self._resolve_runtime_sensitive_source(resolved_split)
        sensitive_series = self._validate_sensitive_series(sensitive, "runtime")
        if sensitive_series is None or len(sensitive_series) != n_rows:
            return None
        if batch_index is not None:
            try:
                aligned = sensitive_series.reindex(batch_index)
                if len(aligned) == n_rows and aligned.notna().all():
                    return aligned.reset_index(drop=True)
            except Exception:
                return None
        return sensitive_series.reset_index(drop=True)

    def _method_accepts_sensitive_features(self, method) -> bool:
        try:
            params = inspect.signature(method).parameters
            if "sensitive_features" in params:
                return True
            return any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            return False

    def _call_with_optional_sensitive(self, method, X, sensitive):
        if sensitive is not None and self._method_accepts_sensitive_features(
            method,
        ):
            return method(X, sensitive_features=sensitive)
        return method(X)

    def _move_torch_model_to_device(self, model_obj, device):
        _ = device
        return model_obj

    def get_model(self) -> BaseEstimator:
        if self._model is None:
            raise ValueError("Model is not fitted yet.")
        if hasattr(self._model, "model"):
            return self._model.model
        return self._model

    def _fit_defended_estimator(self, defended_estimator, data):
        if data is None or not hasattr(defended_estimator, "fit"):
            return defended_estimator

        if getattr(self, "data", None) is None:
            self.data = data

        sensitive = self._resolve_sensitive_features_for_batch(
            data.y_train,
            split="train",
        )
        fit_method = defended_estimator.fit

        # Coerce X_train/y_train to numpy arrays if needed (for fairlearn)
        X = data.X_train
        y = data.y_train

        # Robust shape validation before conversion
        def _check_shape_consistency(arr, name):
            if isinstance(arr, (list, tuple)):
                shapes = [np.shape(v) for v in arr]
                if len(set(shapes)) > 1:
                    raise ValueError(
                        f"Inconsistent shapes in {name}: {shapes}. All elements must have the same shape.",
                    )

        _check_shape_consistency(X, "X_train")
        _check_shape_consistency(y, "y_train")
        if hasattr(X, "numpy"):
            X = X.numpy()
        elif hasattr(X, "detach"):
            X = X.detach().cpu().numpy()
        if hasattr(y, "numpy"):
            y = y.numpy()
        elif hasattr(y, "detach"):
            y = y.detach().cpu().numpy()

        if sensitive is not None and self._method_accepts_sensitive_features(
            fit_method,
        ):
            sensitive_arg = (
                sensitive.to_numpy() if hasattr(sensitive, "to_numpy") else sensitive
            )
            fit_params = self.fit_params if self.fit_params is not None else {}
            fit_method(X, y, sensitive_features=sensitive_arg, **fit_params)
        else:
            fit_params = self.fit_params if self.fit_params is not None else {}
            fit_method(X, y, **fit_params)
        return defended_estimator

    def _resolve_fairlearn_model_param(self, spec, fallback=None):
        if spec is None:
            return fallback
        if hasattr(spec, "get_model") and callable(spec.get_model):
            try:
                return spec.get_model()
            except Exception:
                pass
        if hasattr(spec, "_model") and getattr(spec, "_model", None) is not None:
            return getattr(spec, "_model")
        if isinstance(spec, dict):
            if "model_type" in spec:
                model_type = spec.get("model_type")
                model_params = spec.get("model_params", {}) or {}
                if isinstance(model_type, str):
                    model_obj = load_class(model_type, **model_params)
                    model_obj = self._move_torch_model_to_device(
                        model_obj,
                        spec.get("device", None),
                    )
                    return model_obj
            if "name" in spec:
                spec = {
                    "_target_": spec["name"],
                    **{k: v for k, v in spec.items() if k != "name"},
                }
            if "_target_" in spec:
                target = spec.get("_target_")
                kwargs = {k: v for k, v in spec.items() if k != "_target_"}
                if isinstance(target, str):
                    obj = load_class(target, **kwargs)
                    get_model = getattr(obj, "get_model", None)
                    if callable(get_model):
                        try:
                            return get_model()
                        except Exception:
                            return obj
                    return obj
        if isinstance(spec, ConfigBase):
            get_model = getattr(spec, "get_model", None)
            if callable(get_model):
                return get_model()
            spec_dict = spec.to_dict()
            target = spec_dict.get("model_type")
            params = spec_dict.get("model_params", {})
            if isinstance(target, str):
                return load_class(target, **params)
        if isinstance(spec, str):
            if "." in spec or ":" in spec:
                return load_class(spec)
            return spec
        return spec

    def _resolve_fairness_defense_spec(self):
        if hasattr(self, "defense_name"):
            return getattr(self, "defense_name", None), dict(
                getattr(self, "defense_params", {}) or {},
            )
        defense_obj = getattr(self, "defense", None)
        if defense_obj is not None:
            return getattr(defense_obj, "defense_name", None), dict(
                getattr(defense_obj, "defense_params", {}) or {},
            )
        return None, {}

    def _apply_fairlearn_defense(self, data):
        defense_name, defense_params = self._resolve_fairness_defense_spec()
        if not defense_name or not defense_name.startswith("fairlearn."):
            raise ValueError(
                "Fairlearn defense helper requires a fairlearn defense_name",
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

        start = time.process_time()
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
            if constraints is not None:
                defended_estimator = defense_class(
                    estimator=base_estimator,
                    constraints=constraints,
                    **defense_params,
                )
            else:
                defended_estimator = defense_class(
                    estimator=base_estimator,
                    **defense_params,
                )
        elif fairlearn_submodule == "adversarial":
            predictor_model = self._resolve_fairlearn_model_param(
                defense_params.pop("predictor_model", None),
                fallback=base_estimator,
            )
            predictor_model = FairlearnDefenseConfig._adapt_binary_torch_predictor(
                predictor_model,
                data,
            )
            adversary_model = self._resolve_fairlearn_model_param(
                defense_params.pop("adversary_model", None),
                fallback=base_estimator,
            )
            defense_params["predictor_model"] = predictor_model
            defense_params["adversary_model"] = adversary_model
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
        self.defense_application_time = time.process_time() - start
        return defended_estimator

    def _train(self, X: pd.DataFrame, y: pd.Series):
        if self._model is None:
            raise ValueError("Model not initialized")
        start_time = time.process_time()
        assert hasattr(self._model, "fit"), "Model does not have a fit method"
        fit_params = getattr(self, "fit_params", None) or {}
        sensitive = self._resolve_sensitive_features_for_batch(y, split="train")
        if (
            sensitive is not None
            and self._method_accepts_sensitive_features(self._model.fit)
            and "sensitive_features" not in fit_params
        ):
            fit_params = {**fit_params, "sensitive_features": sensitive}
        self._model.fit(X, y, **fit_params)
        self.training_time = time.process_time() - start_time
        self.training_n = len(y)
        logger.info(f"Model trained in {self.training_time:.2f} seconds")

    def _predict(self, X: pd.DataFrame) -> Any:
        if self._model is None:
            raise ValueError("Model not initialized")
        sensitive = self._resolve_sensitive_features_for_batch(X, split="test")
        try:
            return self._call_with_optional_sensitive(
                self._model.predict,
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

    def _predict_proba(self, X: pd.DataFrame) -> Any:
        if self._model is None:
            raise ValueError("Model not initialized")
        sensitive = self._resolve_sensitive_features_for_batch(X, split="test")
        predict_proba = getattr(self._model, "predict_proba", None)
        if callable(predict_proba):
            return self._call_with_optional_sensitive(
                predict_proba,
                X,
                sensitive,
            )

        # Fallback for torch-style models without predict_proba: derive probabilities
        # from raw outputs or labels so probability-based scorers can run.
        raw_pred = self._predict(X)
        return probabilities_from_model_outputs(raw_pred)

    def __post_init__(self):
        # Auto-select fairness-compatible scorer if not set
        if (
            is_default_config_value(getattr(self, "scorer", None), include_best=False)
            or getattr(self, "scorer", None) is None
        ):
            from deckard.score import (
                DefaultFairlearnClassificationConfig,
                DefaultFairlearnRegressionConfig,
            )

            # Use classifier attribute to determine which default scorer to use
            if hasattr(self, "classifier") and self.classifier is False:
                self.scorer = DefaultFairlearnRegressionConfig()
            else:
                self.scorer = DefaultFairlearnClassificationConfig()
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()


@dataclass(eq=False)
class FairlearnModelConfig(_SensitiveBehaviorMixin, ModelConfig):
    """Fairness-aware model config for sklearn models.

    Inherits sklearn training/prediction from ModelConfig and adds
    fairness-aware scoring and defense support via _FairnessBehaviorMixin.
    """

    data: Union[FairlearnDataConfig, None] = None
    fit_params: dict = field(default_factory=dict)


@dataclass(eq=False)
class FairlearnPytorchModelConfig(_SensitiveBehaviorMixin, PytorchModelConfig):
    """Fairness-aware model config for PyTorch models.

    Inherits all torch training/prediction/defense from PytorchModelConfig
    and adds fairness-aware scoring via _FairnessBehaviorMixin.
    """

    data: Union[FairlearnDataConfig, None] = None

    def _train(self, X, y):
        return PytorchModelConfig._train(self, X, y)

    def _predict(self, X):
        return PytorchModelConfig._predict(self, X)


class BinaryLogitAdapter:
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


@dataclass(eq=False)
class FairlearnDefenseConfig(_SensitiveBehaviorMixin, DefenseConfig):
    """Fairness-aware defense config that inherits DefenseConfig."""

    data: Union[FairlearnDataConfig, None] = None

    def apply_defense(self, data: Any) -> "BaseEstimator":
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
