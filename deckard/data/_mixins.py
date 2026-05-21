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
        if split in {"val", "attack-val"}:
            return getattr(self.data, "_sensitive_val", None)
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
        if sensitive is None:
            sensitive = getattr(data, "_sensitive_train", None)
            if sensitive is not None and hasattr(sensitive, "reset_index"):
                sensitive = sensitive.reset_index(drop=True)
        fit_method = defended_estimator.fit

        X = data.X_train
        y = data.y_train

        # Some fairness datasets emit tuple-like rows (features, label, sensitive).
        # Normalize into homogeneous feature matrix and optional sensitive vector.
        if (
            isinstance(X, (list, tuple))
            and len(X) > 0
            and isinstance(X[0], (list, tuple))
        ):
            rows = list(X)
            X = [row[0] for row in rows]
            if sensitive is None and len(rows[0]) >= 3:
                sensitive = np.asarray([row[2] for row in rows])
        elif (
            isinstance(X, np.ndarray)
            and X.ndim == 2
            and X.shape[1] >= 2
            and X.dtype == object
        ):
            rows = X.tolist()
            X = [row[0] for row in rows]
            if sensitive is None and len(rows[0]) >= 3:
                sensitive = np.asarray([row[2] for row in rows])
        elif hasattr(X, "__len__") and hasattr(X, "__getitem__") and len(X) > 0:
            first_row = None
            if hasattr(X, "iloc"):
                try:
                    first_row = X.iloc[0]
                except Exception:
                    first_row = None
            else:
                try:
                    first_row = X[0]
                except Exception:
                    first_row = None
            if isinstance(first_row, (list, tuple)):
                rows = [X[i] for i in range(len(X))]
                X = [row[0] for row in rows]
                if sensitive is None and len(rows[0]) >= 3:
                    sensitive = np.asarray([row[2] for row in rows])
        self._check_shape_consistency(X, "X_train")
        self._check_shape_consistency(y, "y_train")
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

    def _check_shape_consistency(self, arr, name):
        if isinstance(arr, (list, tuple)):
            shapes = [np.shape(v) for v in arr]
            if len(set(shapes)) > 1:
                raise ValueError(
                    f"Inconsistent shapes in {name}: {shapes}. All elements must have the same shape.",
                )


@dataclass(eq=False, kw_only=True)
class DataPipelineMixin:
    """Reusable pipeline application behavior for data pipeline configs."""

    _PIPELINE_META_KEYS = {
        "name",
        "args",
        "kwargs",
        "fit_X",
        "fit_y",
        "fit_Xy",
        "fit_pre-sample",
        "fit_pre_sample",
        "fit_presample",
        "fit_post-sample",
        "fit_post_sample",
        "fit_postsample",
        "dtype",
        "plugin_hook",
    }

    def normalize_step_hooks(self, raw_hooks: Any) -> list[str]:
        """Normalize ``plugin_hook`` declarations from pipeline step config."""
        if raw_hooks is None:
            return []
        if isinstance(raw_hooks, str):
            hooks = [raw_hooks]
        elif isinstance(raw_hooks, (list, tuple, set)):
            hooks = list(raw_hooks)
        else:
            raise TypeError(
                f"plugin_hook must be None, str, or list-like. Got {type(raw_hooks)}",
            )
        return [str(h).strip().lower() for h in hooks if str(h).strip()]

    def pipeline_declares_hook(self, hook_name: str) -> bool:
        """Return True when any pipeline step declares the requested hook."""
        target_hook = str(hook_name).strip().lower()
        for _, step_config in getattr(self, "pipeline", {}).items():
            hooks = self.normalize_step_hooks(step_config.get("plugin_hook", None))
            if target_hook in hooks:
                return True
        return False

    def build_pipeline(self) -> Any:
        """Build a pipeline object through a framework hook."""
        create_pipeline = getattr(self, "create_pipeline", None)
        if callable(create_pipeline):
            return create_pipeline()
        raise AttributeError(
            f"{type(self).__name__} must define create_pipeline() for pipeline support.",
        )

    def fit_presample(self, X: Any, y: Any) -> tuple:
        """Run pre-sample X-transform pipeline steps on ``X`` when configured."""
        return self._apply_configured_pipeline_stage(X, y, stage="pre_sample")

    def fit_X(self, X: Any, y: Any) -> tuple:
        """Run post-sample X-transform pipeline steps on ``X`` when configured."""
        return self._apply_configured_pipeline_stage(X, y, stage="X")

    def fit_y(self, X: Any, y: Any) -> tuple:
        """Run post-sample y-transform pipeline steps on ``y`` when configured."""
        return self._apply_configured_pipeline_stage(X, y, stage="y")

    def fit_Xy(self, X: Any, y: Any) -> tuple:
        """Run post-sample joint X/y pipeline steps when configured."""
        return self._apply_configured_pipeline_stage(X, y, stage="Xy")

    def run_pipeline(self, pipeline: Any = None) -> Any:
        """Transform runtime payloads using fitted post-sample X/y pipelines."""
        if pipeline is None:
            return None
        if not isinstance(pipeline, tuple) or len(pipeline) != 2:
            return pipeline
        X, y = pipeline
        xy_pipe = getattr(self, "_fitted_pipeline_Xy", None)
        x_pipe = getattr(self, "_fitted_pipeline_X", None)
        y_pipe = getattr(self, "_fitted_pipeline_y", None)
        if xy_pipe is not None:
            X = self._transform_with_pipeline(xy_pipe, X)
        if x_pipe is not None:
            X = self._transform_with_pipeline(x_pipe, X)
        if y_pipe is not None:
            y = self._transform_with_y_pipeline(y_pipe, y)
        return X, y

    def _step_flag(
        self,
        step_config: dict[str, Any],
        flag: str,
        default: bool,
    ) -> bool:
        aliases: dict[str, tuple[str, ...]] = {
            "fit_X": ("fit_X",),
            "fit_y": ("fit_y",),
            "fit_Xy": ("fit_Xy",),
            "fit_pre_sample": (
                "fit_pre-sample",
                "fit_pre_sample",
                "fit_presample",
            ),
            "fit_post_sample": (
                "fit_post-sample",
                "fit_post_sample",
                "fit_postsample",
            ),
        }
        for key in aliases[flag]:
            if key in step_config:
                return bool(step_config[key])
        return default

    def _instantiate_pipeline_step(
        self,
        step_name: str,
        step_config: dict[str, Any],
    ) -> Any:
        class_name = step_config.get("name")
        if not class_name:
            raise ValueError(f"Pipeline step '{step_name}' must define a 'name'")
        args = list(step_config.get("args", []) or [])
        kwargs = dict(step_config.get("kwargs", {}) or {})
        extra_kwargs = {
            k: v
            for k, v in dict(step_config).items()
            if k not in self._PIPELINE_META_KEYS
        }
        kwargs.update(extra_kwargs)
        return load_class(class_name, *args, **kwargs)

    def _collect_stage_steps(
        self,
        stage: str,
    ) -> tuple[list[tuple[str, Any, Any]], list[tuple[str, Any]]]:
        pipeline_cfg = getattr(self, "pipeline", {}) or {}
        if not isinstance(pipeline_cfg, dict):
            return [], []

        x_steps: list[tuple[str, Any, Any]] = []
        y_steps: list[tuple[str, Any]] = []
        for step_name, raw_step_cfg in pipeline_cfg.items():
            if not isinstance(raw_step_cfg, dict):
                continue
            cfg = dict(raw_step_cfg)
            fit_x_explicit = "fit_X" in cfg
            fit_x = self._step_flag(cfg, "fit_X", True)
            fit_y = self._step_flag(cfg, "fit_y", False)
            fit_xy = self._step_flag(cfg, "fit_Xy", False)
            fit_pre = self._step_flag(cfg, "fit_pre_sample", False)
            fit_post = self._step_flag(cfg, "fit_post_sample", True)

            if stage == "pre_sample" and not fit_pre:
                continue
            if stage in {"X", "y", "Xy"} and not fit_post:
                continue
            if stage == "X" and not fit_x:
                continue
            if stage == "X" and fit_xy and not fit_x_explicit:
                continue
            if stage == "y" and not fit_y:
                continue
            if stage == "Xy" and not fit_xy:
                continue

            step_obj = self._instantiate_pipeline_step(step_name, cfg)
            if stage in {"X", "pre_sample"}:
                x_steps.append((step_name, step_obj, cfg.get("dtype", None)))
            elif stage == "y":
                y_steps.append((step_name, step_obj))
            elif stage == "Xy":
                x_steps.append((step_name, step_obj, cfg.get("dtype", None)))
        return x_steps, y_steps

    def _build_x_pipeline(self, x_steps: list[tuple[str, Any, Any]]) -> Any:
        if not x_steps:
            return None
        from sklearn.compose import make_column_selector, make_column_transformer
        from sklearn.pipeline import Pipeline

        typed_steps = [(n, t, d) for n, t, d in x_steps if d is not None]
        if typed_steps:
            transforms = []
            passthrough_steps = []
            for name, transformer, dtype in typed_steps:
                dtype_text = str(dtype).strip().lower()
                if dtype_text in {"num", "numeric", "float", "int"}:
                    selector = make_column_selector(dtype_include=np.number)
                elif dtype_text in {"object", "string", "category"}:
                    selector = make_column_selector(dtype_include=object)
                else:
                    passthrough_steps.append((name, transformer))
                    continue
                transforms.append((transformer, selector))
            if transforms:
                untyped_steps = [
                    (name, transformer)
                    for name, transformer, dtype in x_steps
                    if dtype is None
                ]
                return Pipeline(
                    steps=[
                        (
                            "preprocess",
                            make_column_transformer(
                                *transforms,
                                remainder="passthrough",
                                verbose_feature_names_out=False,
                            ),
                        ),
                        *untyped_steps,
                        *passthrough_steps,
                    ],
                )
        return Pipeline(
            steps=[(name, transformer) for name, transformer, _ in x_steps],
        )

    def _fit_transform_with_pipeline(self, pipeline_obj: Any, X: Any, y: Any) -> Any:
        if pipeline_obj is None:
            return X
        if y is not None:
            pipeline_obj.fit(X, y)
        else:
            pipeline_obj.fit(X)
        return self._transform_with_pipeline(pipeline_obj, X)

    def _transform_with_pipeline(self, pipeline_obj: Any, X: Any) -> Any:
        transformed = pipeline_obj.transform(X)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        if isinstance(X, pd.DataFrame):
            if isinstance(transformed, pd.DataFrame):
                out = transformed.copy()
                out.index = X.index
                return out
            try:
                cols = list(pipeline_obj.get_feature_names_out(X.columns))
            except Exception:
                cols = [
                    f"feature_{i}" for i in range(np.asarray(transformed).shape[1])
                ]
            return pd.DataFrame(transformed, columns=cols, index=X.index)
        return transformed

    def _transform_with_y_pipeline(
        self,
        y_pipeline: list[tuple[str, Any]],
        y: Any,
    ) -> Any:
        y_frame = y.to_frame() if isinstance(y, pd.Series) else pd.DataFrame(y)
        for _, stage in y_pipeline:
            y_frame = stage.transform(y_frame)
            if hasattr(y_frame, "toarray"):
                y_frame = y_frame.toarray()
            if not isinstance(y_frame, pd.DataFrame):
                y_frame = pd.DataFrame(y_frame)
        if y_frame.shape[1] == 1:
            return y_frame.iloc[:, 0]
        return y_frame

    def _fit_transform_y_pipeline(self, y_steps: list[tuple[str, Any]], y: Any) -> Any:
        if not y_steps:
            return y
        y_frame = y.to_frame() if isinstance(y, pd.Series) else pd.DataFrame(y)
        for _, stage in y_steps:
            stage.fit(y_frame)
            y_frame = stage.transform(y_frame)
            if hasattr(y_frame, "toarray"):
                y_frame = y_frame.toarray()
            if not isinstance(y_frame, pd.DataFrame):
                y_frame = pd.DataFrame(y_frame)
        if y_frame.shape[1] == 1:
            return y_frame.iloc[:, 0]
        return y_frame

    def _apply_configured_pipeline_stage(self, X: Any, y: Any, stage: str) -> tuple:
        x_steps, y_steps = self._collect_stage_steps(stage=stage)
        if stage in {"X", "pre_sample"}:
            x_pipeline = self._build_x_pipeline(x_steps)
            if x_pipeline is None:
                return X, y
            X_t = self._fit_transform_with_pipeline(x_pipeline, X, y)
            if stage == "X":
                self._fitted_pipeline_X = x_pipeline
            return X_t, y
        if stage == "y":
            if not y_steps:
                return X, y
            y_t = self._fit_transform_y_pipeline(y_steps, y)
            self._fitted_pipeline_y = y_steps
            return X, y_t
        if stage == "Xy":
            # For joint stages, prefer estimators that accept fit(X, y) and transform(X).
            x_pipeline = self._build_x_pipeline(x_steps)
            if x_pipeline is None:
                return X, y
            X_t = self._fit_transform_with_pipeline(x_pipeline, X, y)
            self._fitted_pipeline_Xy = x_pipeline
            return X_t, y
        return X, y

    def declares_hook(self, hook_name: str) -> bool:
        return self.pipeline_declares_hook(hook_name)

    def apply_to(self, data: Any, mode: Optional[str] = None, **kwargs: Any) -> Any:
        """Apply this pipeline config to a data config instance."""
        _ = (mode, kwargs)
        runtime = copy.copy(self)
        runtime.__dict__.update(data.__dict__)
        runtime.plugins = list(getattr(data, "plugins", []) or [])

        if not hasattr(runtime, "data_load_time") or runtime.data_load_time is None:
            runtime.load_dataset()

        if hasattr(runtime, "run_sampling_with_pipeline_hooks"):
            runtime.run_sampling_with_pipeline_hooks()
        else:
            runtime.fit()

        if hasattr(runtime, "apply_pipeline_behavior"):
            runtime.apply_pipeline_behavior()

        runtime._copy_runtime_state_to(data)
        return data


@dataclass(eq=False, kw_only=True)
class DataLoaderMixin:
    """Reusable data-loading behavior for data configs."""

    def load_raw_data(self) -> Any:
        """Public entry-point for loading raw features and target."""
        return self.load_dataset()


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

    def score(
        self,
        *args: Any,
        mode: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Canonical public entry-point for dataset scoring."""
        return self._score_runtime(*args, mode=mode, **kwargs)


__all__ = [
    "RuntimePayload",
    "_SensitiveColumnsMixin",
    "DataPipelineMixin",
    "DataLoaderMixin",
    "DataPluginRuntimeMixin",
    "DataScoreMixin",
]
