"""Framework-independent data mixins shared across plugin families.

These mixins contain only generic behavior that does not depend on any
optional plugin library (anjana, fairlearn, lifelines, yellowbrick, etc.).
Plugin-specific behavior lives in the corresponding plugin module instead.

This module also defines the shared ``RuntimePayload`` protocol marker used by
plugin runtime call signatures to avoid duplicate local protocol declarations.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union

import numpy as np
import pandas as pd

from ..orchestration import (
    normalize_runtime_split_mode,
    resolve_sensitive_split_payload,
)


class RuntimePayload(Protocol):
    """Central runtime payload marker for plugin/data/mode call signatures.

    This protocol is intentionally opaque and shared across plugin families to
    avoid repeating local marker protocols in each module.
    """


@dataclass(eq=False, kw_only=True)
class SensitiveColumnsMixin:
    """Framework-independent sensitive-column behavior.

    Provides field declarations and helper methods for resolving and
    validating sensitive feature columns. Both the anjana and fairlearn
    plugin families inherit from this mixin so they share a common
    interface without depending on each other.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    sensitive_columns: Optional[Union[str, list]] = None
    fairness_defense: Union[None, bool, Dict[str, Any], list] = None

    def _normalize_sensitive_columns(self) -> list[str]:
        """Return configured sensitive columns as a normalized string list."""
        cols = self.sensitive_columns
        if cols is None:
            return []
        if isinstance(cols, str):
            return [cols]
        return [str(col) for col in cols]

    @staticmethod
    def _resolve_sensitive_column_matches(
        frame_columns: list[str],
        sensitive_column: str,
    ) -> list[str]:
        """Return matching transformed columns for a configured sensitive column."""
        token = str(sensitive_column)
        token_dash = token.replace(".", "-")
        matches: list[str] = []
        for column in frame_columns:
            if (
                column == token
                or column.startswith(f"{token}_")
                or column == token_dash
                or column.startswith(f"{token_dash}_")
            ):
                matches.append(column)
        return matches

    def _sensitive_features_from_frame(
        self,
        frame: Optional[Union[pd.DataFrame, "pd.Series"]],
    ) -> pd.Series | pd.DataFrame:
        """Resolve sensitive payload from frame, preserving one-hot expansions."""
        if frame is None:
            raise ValueError("_sensitive_features_from_frame: frame must not be None")
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)

        cols = self._normalize_sensitive_columns()
        if len(cols) == 0:
            raise ValueError("sensitive_columns must be configured")

        frame_columns = [str(col) for col in frame.columns]
        selected: list[str] = []
        missing: list[str] = []
        for col in cols:
            matches = self._resolve_sensitive_column_matches(frame_columns, col)
            if len(matches) == 0:
                missing.append(col)
                continue
            selected.extend(matches)

        if missing:
            raise KeyError(
                f"Sensitive feature columns not found: {missing}. "
                f"Available columns: {list(frame.columns)}",
            )

        deduped = list(dict.fromkeys(selected))
        if len(deduped) == 1:
            return frame[deduped[0]]
        return frame[deduped].copy()

    def _resolve_post_transform_feature_names(
        self,
        frame: pd.DataFrame,
    ) -> list[str]:
        """Best-effort feature-name ordering after typed preprocessing.

        Mirrors the ordering behavior of ``DataPipeline._build_x_pipeline`` for
        typed steps (numeric/object selectors followed by passthrough columns).
        """
        pipeline_cfg = getattr(self, "pipeline", None)
        if pipeline_cfg is None:
            return list(frame.columns)
        if hasattr(pipeline_cfg, "pipeline"):
            pipeline_cfg = getattr(pipeline_cfg, "pipeline", None)
        if isinstance(pipeline_cfg, Mapping):
            pipeline_cfg = dict(pipeline_cfg)
        if not isinstance(pipeline_cfg, dict):
            return list(frame.columns)

        typed_groups: list[list[str]] = []
        for _, raw_step in pipeline_cfg.items():
            if isinstance(raw_step, Mapping):
                raw_step = dict(raw_step)
            if not isinstance(raw_step, dict):
                continue
            dtype = raw_step.get("dtype", None)
            if dtype is None:
                continue
            dtype_text = str(dtype).strip().lower()
            if dtype_text in {"num", "numeric", "float", "int"}:
                selected = list(frame.select_dtypes(include=np.number).columns)
            elif dtype_text in {"object", "string", "category"}:
                selected = list(
                    frame.select_dtypes(
                        include=["object", "string", "category"],
                    ).columns,
                )
            else:
                continue
            typed_groups.append([str(col) for col in selected])

        if len(typed_groups) == 0:
            return list(frame.columns)

        transformed_order: list[str] = []
        seen: set[str] = set()
        for group in typed_groups:
            for col in group:
                transformed_order.append(col)
                seen.add(col)
        for col in frame.columns:
            col_name = str(col)
            if col_name not in seen:
                transformed_order.append(col_name)
        return transformed_order

    def _resolve_post_transform_feature_names_preview(
        self,
        frame: pd.DataFrame,
    ) -> list[str] | None:
        """Try to infer real post-transform feature names from a fitted preview.

        This is primarily used to resolve sensitive feature ids when typed
        preprocessing (for example object dtype + ``OneHotEncoder``) expands
        source columns into multiple transformed columns.
        """
        pipeline_runtime = getattr(self, "pipeline", None)
        if pipeline_runtime is None:
            return None

        try:
            from .pipeline.base import DataPipeline
        except Exception:
            return None

        if isinstance(pipeline_runtime, DataPipeline):
            pipeline_cfg = dict(pipeline_runtime.pipeline or {})
        elif isinstance(pipeline_runtime, Mapping):
            pipeline_cfg = dict(pipeline_runtime)
        else:
            return None

        preview_runtime = DataPipeline(pipeline=pipeline_cfg)
        x_steps = preview_runtime._collect_x_steps(stage="X")
        if len(x_steps) == 0:
            return list(frame.columns)

        preview_pipeline = preview_runtime._build_x_pipeline(x_steps)
        if preview_pipeline is None:
            return list(frame.columns)

        y_fit = getattr(self, "_y", None)
        try:
            if (
                y_fit is not None
                and hasattr(y_fit, "__len__")
                and len(y_fit) == len(frame)
            ):
                preview_pipeline.fit(frame, y_fit)
            else:
                preview_pipeline.fit(frame)
        except Exception:
            return None

        try:
            names = list(preview_pipeline.get_feature_names_out(frame.columns))
            return [str(name) for name in names]
        except Exception:
            try:
                transformed = preview_pipeline.transform(frame)
            except Exception:
                return None
            width = np.asarray(transformed).shape[1]
            return [f"feature_{i}" for i in range(width)]

    def _pipeline_uses_typed_preprocessing(self) -> bool:
        """Return True when pipeline config includes dtype-targeted X steps."""
        pipeline_cfg = getattr(self, "pipeline", None)
        if pipeline_cfg is None:
            return False
        if hasattr(pipeline_cfg, "pipeline"):
            pipeline_cfg = getattr(pipeline_cfg, "pipeline", None)
        if isinstance(pipeline_cfg, Mapping):
            pipeline_cfg = dict(pipeline_cfg)
        if not isinstance(pipeline_cfg, dict):
            return False
        for _, raw_step in pipeline_cfg.items():
            if isinstance(raw_step, Mapping):
                raw_step = dict(raw_step)
            if not isinstance(raw_step, dict):
                continue
            dtype = raw_step.get("dtype", None)
            if dtype is None:
                continue
            dtype_text = str(dtype).strip().lower()
            if dtype_text in {
                "num",
                "numeric",
                "float",
                "int",
                "object",
                "string",
                "category",
            }:
                return True
        return False

    def _resolve_sensitive_feature_ids_for_pipeline(
        self,
        frame: pd.DataFrame,
    ):
        """Resolve sensitive feature identifiers compatible with pipeline runtime.

        Returns column names for direct DataFrame flows and integer indices when
        typed preprocessing is configured (post-transform pipeline internals use
        numpy arrays where fairlearn expects positional feature ids).
        """
        cols = self._normalize_sensitive_columns()
        if len(cols) == 0:
            raise ValueError("sensitive_columns must be configured")

        missing_input = [col for col in cols if col not in frame.columns]
        if missing_input:
            raise RuntimeError(
                f"Sensitive features not found for {cols}.",
            )

        post_columns = self._resolve_post_transform_feature_names(frame)
        uses_typed_preprocess = self._pipeline_uses_typed_preprocessing()
        if not uses_typed_preprocess:
            return list(cols)

        preview_columns = self._resolve_post_transform_feature_names_preview(frame)
        if preview_columns is not None and len(preview_columns) > 0:
            post_columns = preview_columns

        resolved_indices: list[int] = []
        unresolved: list[str] = []
        for col in cols:
            matches = [
                idx
                for idx, name in enumerate(post_columns)
                if name == col or name.startswith(f"{col}_")
            ]
            if len(matches) == 0:
                unresolved.append(col)
                continue
            resolved_indices.extend(matches)

        if unresolved:
            raise RuntimeError(
                "Sensitive features not found in transformed feature space for "
                f"{unresolved}. Available transformed columns: {post_columns}",
            )
        return sorted(set(resolved_indices))

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
        cols = self._normalize_sensitive_columns()
        if len(cols) == 0:
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
                matches = self._resolve_sensitive_column_matches(
                    [str(c) for c in frame.columns],
                    cols[0],
                )
                if len(matches) > 0:
                    if len(matches) == 1:
                        return frame[matches[0]].astype(str)
                    labels_df = frame[matches].astype(str)
                    return labels_df.apply(
                        lambda row: tuple(row.values.tolist()),
                        axis=1,
                    )
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
        sensitive: pd.Series | pd.DataFrame | Any,
        context: str,
    ) -> pd.Series | pd.DataFrame:
        """Validate that *sensitive* is non-empty and non-null."""
        if isinstance(sensitive, pd.DataFrame):
            if len(sensitive) == 0 or sensitive.shape[1] == 0:
                raise ValueError(f"Sensitive features are empty during {context}")
            if sensitive.dropna(how="all").empty:
                raise ValueError(
                    f"Sensitive features are all null during {context}",
                )
            if sensitive.astype(str).apply(lambda c: c.str.strip().eq("")).all().all():
                raise ValueError(f"Sensitive features are blank during {context}")
            return sensitive.reset_index(drop=True)

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
        return resolve_sensitive_split_payload(
            getattr(self, "data", None),
            split,
            aliases={"attack-val": "val"},
        )

    def _resolve_scoring_split(self, mode: str) -> str:
        return normalize_runtime_split_mode(
            mode,
            aliases={"attack": "test", "attack-val": "val"},
        )

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
        sensitive_payload = self._validate_sensitive_series(sensitive, "runtime")
        if sensitive_payload is None or len(sensitive_payload) != n_rows:
            return None
        if batch_index is not None:
            try:
                aligned = sensitive_payload.reindex(batch_index)
                if isinstance(aligned, pd.DataFrame):
                    valid = len(aligned) == n_rows and aligned.notna().all().all()
                else:
                    valid = len(aligned) == n_rows and aligned.notna().all()
                if valid:
                    return aligned.reset_index(drop=True)
            except Exception:
                return None
        if hasattr(sensitive_payload, "reset_index"):
            return sensitive_payload.reset_index(drop=True)
        return sensitive_payload

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

    def _extract_rows_from_runtime_payload(self, X):
        if (
            isinstance(X, (list, tuple))
            and len(X) > 0
            and isinstance(X[0], (list, tuple))
        ):
            return list(X)
        if (
            isinstance(X, np.ndarray)
            and X.ndim == 2
            and X.shape[1] >= 2
            and X.dtype == object
        ):
            return X.tolist()
        if not (hasattr(X, "__len__") and hasattr(X, "__getitem__") and len(X) > 0):
            return None

        first_row = None
        iloc_accessor = getattr(X, "iloc", None)
        if iloc_accessor is not None:
            try:
                first_row = iloc_accessor[0]
            except Exception:
                first_row = None
        else:
            try:
                first_row = X[0]
            except Exception:
                first_row = None

        if isinstance(first_row, (list, tuple)):
            return [X[i] for i in range(len(X))]
        return None

    def _normalize_training_X_sensitive(self, X, sensitive):
        rows = self._extract_rows_from_runtime_payload(X)
        if not rows:
            return X, sensitive
        normalized_X = [row[0] for row in rows]
        if sensitive is None and len(rows[0]) >= 3:
            sensitive = np.asarray([row[2] for row in rows])
        return normalized_X, sensitive

    def get_model(self) -> RuntimePayload:
        """Return fitted model object, unwrapping wrapper attributes when needed.

        Returns:
            Fitted model runtime payload.

        Raises:
            ValueError: If runtime model has not been fitted.
        """
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
        X, sensitive = self._normalize_training_X_sensitive(X, sensitive)
        self._check_shape_consistency(X, "X_train")
        self._check_shape_consistency(y, "y_train")
        X_payload: Any = X
        y_payload: Any = y
        if hasattr(X_payload, "numpy"):
            X_payload = X_payload.numpy()
        elif hasattr(X_payload, "detach"):
            X_payload = X_payload.detach().cpu().numpy()
        if hasattr(y_payload, "numpy"):
            y_payload = y_payload.numpy()
        elif hasattr(y_payload, "detach"):
            y_payload = y_payload.detach().cpu().numpy()

        fit_params = getattr(self, "fit_params", None)
        fit_params = fit_params if fit_params is not None else {}
        if sensitive is not None and self._method_accepts_sensitive_features(
            fit_method,
        ):
            to_numpy = getattr(sensitive, "to_numpy", None)
            sensitive_arg = to_numpy() if callable(to_numpy) else sensitive
            fit_method(
                X_payload,
                y_payload,
                sensitive_features=sensitive_arg,
                **fit_params,
            )
        else:
            fit_method(X_payload, y_payload, **fit_params)
        return defended_estimator

    def _check_shape_consistency(self, arr, name):
        if isinstance(arr, (list, tuple)):
            shapes = [np.shape(v) for v in arr]
            if len(set(shapes)) > 1:
                raise ValueError(
                    f"Inconsistent shapes in {name}: {shapes}. All elements must have the same shape.",
                )


__all__ = [
    "RuntimePayload",
    "SensitiveColumnsMixin",
]
