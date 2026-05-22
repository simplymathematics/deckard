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

from ..utils import load_class


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



__all__ = [
    "RuntimePayload",
    "_SensitiveColumnsMixin",
    "DataLoaderMixin",
]
