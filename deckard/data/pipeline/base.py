"""Core data pipeline runtime and config markers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from ...utils import load_class

if TYPE_CHECKING:
    from ..base import DataConfig, DataPipelineConfig


@dataclass(eq=False, kw_only=True)
class DataPipeline:
    """Runtime pipeline executor for host data configs.

    Execution order in ``__call__``:
    1) fit_pre_sample
    2) fit_X
    3) fit_y
    4) fit_Xy
    """

    pipeline: dict[str, Any] = field(default_factory=dict)

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

    def __call__(self, host: "DataConfig") -> "DataConfig":
        if getattr(host, "_X", None) is None or getattr(host, "_y", None) is None:
            host.load_dataset()

        self.fit_pre_sample(host)

        if getattr(host, "train_indices", None) is None:
            host._split_loaded_data(run_hooks=True)

        self.fit_X(host)
        self.fit_y(host)
        self.fit_Xy(host)
        return host

    def fit_pre_sample(self, host: "DataConfig") -> None:
        self._run_stage_hooks(host, "before", "fit_pre_sample", "pre-sample")
        x_steps = self._collect_x_steps(stage="pre_sample")
        pipeline = self._build_x_pipeline(x_steps)
        if pipeline is not None and getattr(host, "_X", None) is not None:
            host._X = self._fit_transform_features(
                pipeline,
                X_fit=host._X,
                X_apply=host._X,
                y_fit=getattr(host, "_y", None),
            )
        self._run_stage_hooks(host, "after", "fit_pre_sample", "pre-sample")

    def fit_X(self, host: "DataConfig") -> None:
        self._run_stage_hooks(host, "before", "fit_X", "post-pipeline")
        x_steps = self._collect_x_steps(stage="X")
        pipeline = self._build_x_pipeline(x_steps)
        if pipeline is not None and getattr(host, "X_train", None) is not None:
            host.X_train = self._fit_transform_features(
                pipeline,
                X_fit=host.X_train,
                X_apply=host.X_train,
                y_fit=getattr(host, "y_train", None),
            )
            host.X_test = self._transform_features(pipeline, host.X_test)
            if getattr(host, "X_val", None) is not None:
                host.X_val = self._transform_features(pipeline, host.X_val)
        self._run_stage_hooks(host, "after", "fit_X", "post-pipeline")

    def fit_y(self, host: "DataConfig") -> None:
        self._run_stage_hooks(host, "before", "fit_y", "post-pipeline")
        y_steps = self._collect_y_steps(stage="y")
        if len(y_steps) > 0 and getattr(host, "y_train", None) is not None:
            host.y_train = self._fit_transform_target(y_steps, host.y_train)
            host.y_test = self._transform_target(y_steps, host.y_test)
            if getattr(host, "y_val", None) is not None:
                host.y_val = self._transform_target(y_steps, host.y_val)
        self._run_stage_hooks(host, "after", "fit_y", "post-pipeline")

    def fit_Xy(self, host: "DataConfig") -> None:
        self._run_stage_hooks(host, "before", "fit_Xy", "post-pipeline")
        xy_steps = self._collect_x_steps(stage="Xy")
        pipeline = self._build_x_pipeline(xy_steps)
        if pipeline is not None and getattr(host, "X_train", None) is not None:
            X_parts = [host.X_train, host.X_test]
            y_parts = [host.y_train, host.y_test]
            sizes = [len(host.X_train), len(host.X_test)]
            has_val = getattr(host, "X_val", None) is not None and getattr(host, "y_val", None) is not None
            if has_val:
                X_parts.append(host.X_val)
                y_parts.append(host.y_val)
                sizes.append(len(host.X_val))

            X_all = pd.concat(X_parts, ignore_index=True)
            y_all = pd.concat([pd.Series(part) for part in y_parts], ignore_index=True)
            X_all_t = self._fit_transform_features(
                pipeline,
                X_fit=X_all,
                X_apply=X_all,
                y_fit=y_all,
            )
            host._X = X_all_t
            host._y = y_all
            host._split_loaded_data(run_hooks=False)
        self._run_stage_hooks(host, "after", "fit_Xy", "post-pipeline")

    def _run_stage_hooks(
        self,
        host: "DataConfig",
        when: str,
        stage_name: str,
        score_stage: str,
    ) -> None:
        event = str(when).strip().lower()
        if event not in {"before", "after"}:
            raise ValueError(f"Invalid stage hook event: {when}")

        if hasattr(host, "_run_plugin_hook"):
            host._run_plugin_hook(f"{event}_{stage_name}", stage=stage_name)
            host._run_plugin_hook(f"{event}_pipeline", stage=stage_name)

        if hasattr(host, "_run_score_stage_hooks"):
            host._run_score_stage_hooks(event, score_stage, pipeline_stage=stage_name)

    def _step_flag(self, step_config: dict[str, Any], flag: str, default: bool) -> bool:
        aliases: dict[str, tuple[str, ...]] = {
            "fit_X": ("fit_X",),
            "fit_y": ("fit_y",),
            "fit_Xy": ("fit_Xy",),
            "fit_pre_sample": ("fit_pre-sample", "fit_pre_sample", "fit_presample"),
            "fit_post_sample": ("fit_post-sample", "fit_post_sample", "fit_postsample"),
        }
        for key in aliases[flag]:
            if key in step_config:
                return bool(step_config[key])
        return default

    def _instantiate_pipeline_step(self, step_name: str, step_config: dict[str, Any]) -> Any:
        class_name = step_config.get("name")
        if not class_name:
            raise ValueError(f"Pipeline step '{step_name}' must define a 'name'")
        args = list(step_config.get("args", []) or [])
        kwargs = dict(step_config.get("kwargs", {}) or {})
        extras = {k: v for k, v in dict(step_config).items() if k not in self._PIPELINE_META_KEYS}
        kwargs.update(extras)
        return load_class(class_name, *args, **kwargs)

    def _collect_x_steps(self, stage: str) -> list[tuple[str, Any, Any]]:
        steps: list[tuple[str, Any, Any]] = []
        if not isinstance(self.pipeline, dict):
            return steps
        for step_name, raw_step in self.pipeline.items():
            if not isinstance(raw_step, dict):
                continue
            cfg = dict(raw_step)
            fit_x_explicit = "fit_X" in cfg
            fit_x = self._step_flag(cfg, "fit_X", True)
            fit_xy = self._step_flag(cfg, "fit_Xy", False)
            fit_pre = self._step_flag(cfg, "fit_pre_sample", False)
            fit_post = self._step_flag(cfg, "fit_post_sample", True)

            if stage == "pre_sample" and not fit_pre:
                continue
            if stage in {"X", "Xy"} and not fit_post:
                continue
            if stage == "X" and not fit_x:
                continue
            if stage == "X" and fit_xy and not fit_x_explicit:
                continue
            if stage == "Xy" and not fit_xy:
                continue

            step_obj = self._instantiate_pipeline_step(step_name, cfg)
            steps.append((step_name, step_obj, cfg.get("dtype", None)))
        return steps

    def _collect_y_steps(self, stage: str) -> list[tuple[str, Any]]:
        if stage != "y" or not isinstance(self.pipeline, dict):
            return []
        steps: list[tuple[str, Any]] = []
        for step_name, raw_step in self.pipeline.items():
            if not isinstance(raw_step, dict):
                continue
            cfg = dict(raw_step)
            fit_y = self._step_flag(cfg, "fit_y", False)
            fit_post = self._step_flag(cfg, "fit_post_sample", True)
            if fit_y and fit_post:
                step_obj = self._instantiate_pipeline_step(step_name, cfg)
                steps.append((step_name, step_obj))
        return steps

    def _build_x_pipeline(self, x_steps: list[tuple[str, Any, Any]]) -> Pipeline | None:
        if len(x_steps) == 0:
            return None

        typed_steps = [(n, t, d) for n, t, d in x_steps if d is not None]
        if len(typed_steps) > 0:
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
                transforms.append((name, transformer, selector))
            if len(transforms) > 0:
                untyped = [(n, t) for n, t, d in x_steps if d is None]
                return Pipeline(
                    steps=[
                        (
                            "preprocess",
                            ColumnTransformer(
                                transformers=transforms,
                                remainder="passthrough",
                                verbose_feature_names_out=False,
                            ),
                        ),
                        *untyped,
                        *passthrough_steps,
                    ],
                )

        return Pipeline(steps=[(name, transformer) for name, transformer, _ in x_steps])

    def _fit_transform_features(self, pipeline: Pipeline, X_fit: Any, X_apply: Any, y_fit: Any = None) -> Any:
        if y_fit is not None:
            pipeline.fit(X_fit, y_fit)
        else:
            pipeline.fit(X_fit)
        return self._transform_features(pipeline, X_apply)

    def _transform_features(self, pipeline: Pipeline, X: Any) -> Any:
        transformed = pipeline.transform(X)
        if isinstance(transformed, csr_matrix):
            transformed = transformed.toarray()
        if isinstance(X, pd.DataFrame):
            try:
                cols = list(pipeline.get_feature_names_out(X.columns))
            except Exception:
                cols = [f"feature_{i}" for i in range(np.asarray(transformed).shape[1])]
            return pd.DataFrame(transformed, columns=cols, index=X.index)
        return transformed

    def _fit_transform_target(self, y_steps: list[tuple[str, Any]], y: Any) -> Any:
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

    def _transform_target(self, y_steps: list[tuple[str, Any]], y: Any) -> Any:
        y_frame = y.to_frame() if isinstance(y, pd.Series) else pd.DataFrame(y)
        for _, stage in y_steps:
            y_frame = stage.transform(y_frame)
            if hasattr(y_frame, "toarray"):
                y_frame = y_frame.toarray()
            if not isinstance(y_frame, pd.DataFrame):
                y_frame = pd.DataFrame(y_frame)
        if y_frame.shape[1] == 1:
            return y_frame.iloc[:, 0]
        return y_frame


# Compatibility marker configs still exposed through data.pipeline package.
from ..base import DataPipelineConfig 


@dataclass(eq=False, kw_only=True)
class DefaultDataPipelineConfig(DataPipelineConfig):
    """Default no-op data pipeline config."""

    pipeline: dict = field(default_factory=dict)


@dataclass(eq=False, kw_only=True)
class AnjanaDataPipelineConfig(DataPipelineConfig):
    """Pipeline config marker for anjana-family data flows."""

    pipeline: dict = field(default_factory=dict)


@dataclass(eq=False, kw_only=True)
class FairlearnDataPipelineConfig(DataPipelineConfig):
    """Pipeline config marker for fairlearn-family data flows."""

    pipeline: dict = field(default_factory=dict)


__all__ = [
    "DataPipeline",
    "DataPipelineConfig",
    "DefaultDataPipelineConfig",
    "AnjanaDataPipelineConfig",
    "FairlearnDataPipelineConfig",
]
