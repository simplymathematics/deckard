"""Framework-independent data mixins shared across plugin families.

These mixins contain only generic behavior that does not depend on any
optional plugin library (anjana, fairlearn, lifelines, yellowbrick, etc.).
Plugin-specific behavior lives in the corresponding plugin module instead.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import pandas as pd

from ..frameworks import DataPipelineContractMixin
from ..utils import instantiate_plugin_spec, load_class, normalize_plugin_specs


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


@dataclass(eq=False, kw_only=True)
class DataPipelineMixin(DataPipelineContractMixin):
    """Reusable pipeline application behavior for data pipeline configs."""

    def declares_hook(self, hook_name: str) -> bool:
        return hasattr(self, "_pipeline_declares_hook") and self._pipeline_declares_hook(
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
            raise TypeError(f"Composed sampler must be callable, got {type(sampler_obj)}")
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
    "_SensitiveColumnsMixin",
    "DataPipelineMixin",
    "DataLoaderMixin",
    "DataSamplerMixin",
    "DataPluginRuntimeMixin",
    "DataScoreMixin",
]
