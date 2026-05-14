"""Framework-independent data mixins shared across plugin families.

These mixins contain only generic behavior that does not depend on any
optional plugin library (anjana, fairlearn, lifelines, yellowbrick, etc.).
Plugin-specific behavior lives in the corresponding plugin module instead.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import pandas as pd


@dataclass(eq=False)
class _SensitiveColumnsMixin:
    """Framework-independent sensitive-column behavior.

    Provides field declarations and helper methods for resolving and
    validating sensitive feature columns.  Both the anjana and fairlearn
    plugin families inherit from this mixin so they share a common
    interface without depending on each other.

    Fields
    ------
    sensitive_columns : str | list[str] | None
        Name(s) of the sensitive-feature column(s) in the loaded dataset.
    fairness_defense : dict | list | bool | None
        Defense step specification consumed by fairlearn-aware subclasses.
        Declared here so that ``_PrivacyBehaviorMixin._before_post_init``
        can normalise the value without importing from ``data.fairness``.
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
