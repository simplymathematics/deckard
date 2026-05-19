"""Framework-independent data mixins shared across plugin families.

These mixins contain only generic behavior that does not depend on any
optional plugin library (anjana, fairlearn, lifelines, yellowbrick, etc.).
Plugin-specific behavior lives in the corresponding plugin module instead.

This module also defines the shared ``RuntimePayload`` protocol marker used by
plugin runtime call signatures to avoid duplicate local protocol declarations.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import pandas as pd

from ..frameworks import DataPipelineContractMixin
from ..utils import instantiate_plugin_spec, load_class, normalize_plugin_specs


class RuntimePayload(Protocol):
    """Central runtime payload marker for plugin/data/mode call signatures.

    This protocol is intentionally opaque and shared across plugin families to
    avoid repeating local marker protocols in each module.
    """


@dataclass(eq=False, kw_only=True)
class _SensitiveColumnsMixin:
    """Framework-independent sensitive-column behavior.

    Provides field declarations and helper methods for resolving and
    validating sensitive feature columns. Both the anjana and fairlearn
    plugin families inherit from this mixin so they share a common
    interface without depending on each other.
    """

    sensitive_columns: Optional[Union[str, list]] = None
    fairness_defense: Union[None, bool, Dict[str, Any], list] = None

    def _sensitive_labels_from_targets(
        self,
        frame: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Fallback sensitive labels from y-splits when sensitive feature is a target label."""
        if getattr(self, "X_train", None) is frame:
            y_values = getattr(self, "y_train", None)
        elif getattr(self, "X_test", None) is frame:
            y_values = getattr(self, "y_test", None)
        elif getattr(self, "X_val", None) is frame:
            y_values = getattr(self, "y_val", None)
        elif getattr(self, "_X", None) is frame:
            y_values = getattr(self, "_y", None)
        else:
            y_values = None
        if y_values is None:
            return None
        return pd.Series(y_values).astype(str)

    def _sensitive_labels_from_frame(
        self,
        frame: Optional[Union[pd.DataFrame, "pd.Series"]],
    ) -> pd.Series:
        """Resolve sensitive labels from ``frame`` using ``sensitive_columns``."""
        if frame is None:
            raise ValueError(
                "_sensitive_labels_from_frame: frame must not be None",
            )
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        cols = self.sensitive_columns
        if isinstance(cols, str):
            cols = [cols]
        if cols is None:
            fallback = self._sensitive_labels_from_targets(frame)
            if fallback is not None:
                return fallback
            raise ValueError("sensitive_columns must be configured")
        missing_cols = [col for col in cols if col not in frame.columns]
        if missing_cols:
            if len(cols) == 1:
                fallback = self._sensitive_labels_from_targets(frame)
                if fallback is not None:
                    return fallback
            raise KeyError(
                f"Sensitive feature columns not found: {missing_cols}. "
                f"Available columns: {list(frame.columns)}",
            )
        if len(cols) == 1:
            return frame[cols[0]].astype(str)
        labels_df = frame[cols].astype(str)
        return labels_df.apply(lambda row: tuple(row.values.tolist()), axis=1)

    def _validate_sensitive_runtime(
        self,
        sensitive: pd.Series,
        context: str,
    ) -> pd.Series:
        """Validate that *sensitive* is non-empty and non-null."""
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

    def _resolve_runtime_sensitive_source(self, split: str):
        if getattr(self, "data", None) is None:
            return None
        if split == "train":
            return getattr(self.data, "sensitive_train", None)
        if split == "test":
            return getattr(self.data, "sensitive_test", None)
        if split == "all":
            return getattr(self.data, "sensitive_all", None)
        if split == "val":
            return getattr(self.data, "sensitive_val", None)
        raise ValueError(f"Unsupported fairness split: {split}")

    def _resolve_scoring_split(self, mode: str) -> str:
        if mode == "train":
            return "train"
        if mode in {"test", "attack"}:
            return "test"
        if mode in {"val", "attack-val"}:
            raise NotImplementedError(
                "Validation fairness scoring is not implemented yet",
            )
        if mode == "all":
            return "all"
        raise ValueError(f"Unsupported fairness scoring mode: {mode}")

    def _validate_sensitive_series(self, sensitive, context: str):
        if sensitive is None:
            return None
        return self._validate_sensitive_runtime(sensitive, context)

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

    def get_model(self):
        if getattr(self, "_model", None) is None:
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

        X = data.X_train
        y = data.y_train

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

        fit_params = getattr(self, "fit_params", None)
        fit_params = fit_params if fit_params is not None else {}
        if sensitive is not None and self._method_accepts_sensitive_features(
            fit_method,
        ):
            sensitive_arg = (
                sensitive.to_numpy() if hasattr(sensitive, "to_numpy") else sensitive
            )
            fit_method(X, y, sensitive_features=sensitive_arg, **fit_params)
        else:
            fit_method(X, y, **fit_params)
        return defended_estimator


@dataclass(eq=False, kw_only=True)
class DataPipelineMixin(DataPipelineContractMixin):
    """Reusable pipeline application behavior for data pipeline configs."""

    def declares_hook(self, hook_name: str) -> bool:
        return hasattr(
            self, "_pipeline_declares_hook"
        ) and self._pipeline_declares_hook(
            hook_name,
        )

    def apply_to(self, data: Any, mode: Optional[str] = None, **kwargs: Any) -> Any:
        """Apply this pipeline config to a data config instance."""
        _ = (mode, kwargs)
        runtime = copy.copy(self)
        runtime.__dict__.update(data.__dict__)
        runtime.plugins = list(getattr(data, "plugins", []) or [])

        if not hasattr(runtime, "data_load_time") or runtime.data_load_time is None:
            runtime._load_data()

        if hasattr(runtime, "run_sampling_with_pipeline_hooks"):
            runtime.run_sampling_with_pipeline_hooks()
        else:
            runtime._sample()

        if hasattr(runtime, "apply_pipeline_behavior"):
            runtime.apply_pipeline_behavior()

        runtime._copy_runtime_state_to(data)
        return data


@dataclass(eq=False, kw_only=True)
class DataLoaderMixin:
    """Reusable data-loading behavior for data configs."""

    def load_raw_data(self) -> Any:
        """Public entry-point for loading raw features and target."""
        return self._load_data()


@dataclass(eq=False, kw_only=True)
class DataSamplerMixin:
    """Reusable sampler orchestration behavior for data configs."""

    def split_data(self, run_hooks: bool = True) -> Any:
        """Public entry-point for sampling/splitting loaded data."""
        return self._sample(run_hooks=run_hooks)

    def _resolve_sample(self):
        """Instantiate and return the sampler object."""
        from .sample import KFoldSampler, ShuffleSampler, SplitSampler

        sampler_aliases = {
            "split": SplitSampler,
            "kfold": KFoldSampler,
            "shuffle": ShuffleSampler,
        }

        if isinstance(self.sample, str):
            key = self.sample.lower()
            if key not in sampler_aliases:
                raise ValueError(
                    f"Unknown sampler '{self.sample}'. Must be one of {list(sampler_aliases)}.",
                )
            return sampler_aliases[key]()

        spec = self.sample

        try:
            from omegaconf import DictConfig, OmegaConf

            if isinstance(spec, DictConfig):
                spec = OmegaConf.to_container(spec, resolve=True)
        except ImportError:
            pass

        if isinstance(spec, dict):
            if not spec:
                return None
            spec = dict(spec)
            class_path = spec.pop("name", spec.pop("_target_", None))
            if class_path is None:
                raise ValueError("sample dict must include 'name' or '_target_'")
            return load_class(class_path, **spec)

        if callable(spec) and not isinstance(spec, type):
            return spec

        if isinstance(spec, type):
            return spec()

        raise ValueError(f"Unsupported sample specification: {type(spec)}")

    def compose_sampling_behavior(self):
        """Compose and return the sampler runtime callable used by split/sample flows."""
        sampler_obj = self._resolve_sample()
        if sampler_obj is None:
            from .sample import SplitSampler

            sampler_obj = SplitSampler()
        if not callable(sampler_obj):
            raise TypeError(
                f"Composed sampler must be callable, got {type(sampler_obj)}"
            )
        return sampler_obj


@dataclass(eq=False, kw_only=True)
class DataPluginRuntimeMixin:
    """Reusable plugin orchestration and runtime-state copy behavior."""

    def _copy_runtime_state_to(self, target: Any) -> None:
        runtime_fields = [
            "score_dict",
            "data_load_time",
            "data_sample_time",
            "_X",
            "_y",
            "train_indices",
            "test_indices",
            "val_indices",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "X_val",
            "y_val",
            "train_n",
            "test_n",
            "val_n",
            "pipeline_fit_n",
            "pipeline_transform_n",
            "pipeline_fit_time",
            "pipeline_transform_time",
            "pipeline_y_fit_n",
            "pipeline_y_fit_time",
            "pipeline_y_transform_n",
            "pipeline_y_transform_time",
        ]
        for attr in runtime_fields:
            if hasattr(self, attr):
                setattr(target, attr, getattr(self, attr, None))

    def _instantiate_plugin(self, plugin_spec: Any):
        return instantiate_plugin_spec(plugin_spec, loader=load_class)

    def _get_plugins(self) -> list:
        if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
            plugin_specs = normalize_plugin_specs(self.plugins)
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs: Any) -> list[Any]:
        """Execute one plugin hook across all instantiated plugins."""
        hook_outputs = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs


@dataclass(eq=False, kw_only=True)
class DataScoreMixin:
    """Reusable score-wrapper behavior for data configs."""

    def compute_score(
        self,
        *args: Any,
        mode: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Public entry-point for dataset scoring."""
        return self._score(*args, mode=mode, **kwargs)


__all__ = [
    "RuntimePayload",
    "_SensitiveColumnsMixin",
    "DataPipelineMixin",
    "DataLoaderMixin",
    "DataSamplerMixin",
    "DataPluginRuntimeMixin",
    "DataScoreMixin",
]
