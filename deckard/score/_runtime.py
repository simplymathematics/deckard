from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def series_like_to_float_dict(values: Any) -> dict[str, float]:
    """Flatten scalar/Series/DataFrame-like values into score key/value pairs."""
    if isinstance(values, dict):
        flattened: dict[str, float] = {}

        def _flatten(prefix: str, payload: Any) -> None:
            if isinstance(payload, dict):
                for key, value in payload.items():
                    child_key = str(key).strip()
                    next_prefix = (
                        child_key
                        if prefix == ""
                        else f"{prefix}_{child_key}"
                    )
                    _flatten(next_prefix, value)
                return
            if isinstance(payload, pd.Series):
                for key, value in payload.items():
                    child_key = str(key).strip()
                    next_prefix = (
                        child_key
                        if prefix == ""
                        else f"{prefix}_{child_key}"
                    )
                    _flatten(next_prefix, value)
                return
            if isinstance(payload, pd.DataFrame):
                df_payload = series_like_to_float_dict(payload)
                for key, value in df_payload.items():
                    next_prefix = key if prefix == "" else f"{prefix}_{key}"
                    flattened[next_prefix] = float(value)
                return
            key = prefix if prefix != "" else "value"
            scalar = np.asarray(payload)
            if scalar.ndim == 0:
                flattened[key] = float(scalar)
                return
            raise TypeError(
                "Score dictionary values must be scalar-like after flattening; "
                f"got shape {scalar.shape} at key '{key}'.",
            )

        _flatten("", values)
        return flattened

    if isinstance(values, pd.DataFrame):
        flattened: dict[str, float] = {}
        for row_key, row_values in values.to_dict(orient="index").items():
            if isinstance(row_key, tuple):
                row_label = "_".join(str(group) for group in row_key)
            else:
                row_label = str(row_key)
            for col_key, col_val in row_values.items():
                flattened[f"{row_label}_{col_key}"] = float(col_val)
        return flattened
    if isinstance(values, pd.Series):
        return {str(key): float(value) for key, value in values.items()}
    scalar = np.asarray(values)
    if scalar.ndim == 0:
        return {"value": float(scalar)}
    raise TypeError(
        "Score values must be scalar, dict, pandas Series, or pandas DataFrame; "
        f"got shape {scalar.shape}.",
    )


def resolve_yt_yp(
    mode: str | None,
    data: Any,
    model: Any,
    attack: Any,
    y_pred: Any,
    y_true: Any,
) -> tuple[Any, Any]:
    """Resolve ``y_true`` and ``y_pred`` from mode and runtime context."""
    if y_pred is not None:
        return y_true, y_pred
    if mode == "test":
        if data is not None:
            y_true = getattr(data, "y_test", y_true)
        if model is not None:
            y_pred = getattr(model, "test_predictions", None)
            if y_pred is None:
                y_pred = getattr(model, "predictions", None)
    elif mode == "train":
        if data is not None:
            y_true = getattr(data, "y_train", y_true)
        if model is not None:
            y_pred = getattr(model, "training_predictions", None)
    elif mode == "attack":
        if data is not None and attack is not None:
            y_test = np.asarray(getattr(data, "y_test", y_true))
            attack_size = getattr(attack, "attack_size", None)
            y_true = y_test[:attack_size] if attack_size is not None else y_test
        if attack is not None:
            y_pred = getattr(attack, "attack_predictions", None)
    elif mode == "val":
        if data is not None:
            y_true = getattr(data, "y_val", y_true)
        if model is not None:
            y_pred = getattr(model, "val_predictions", None)
    elif mode == "attack-val":
        if data is not None:
            y_true = getattr(data, "y_val", y_true)
        if attack is not None:
            y_pred = getattr(attack, "attack_predictions", None)
    elif mode == "pre-sample":
        if data is not None:
            y_true = getattr(data, "y", getattr(data, "_y", y_true))
            y_pred = getattr(data, "X", getattr(data, "_X", y_pred))
    return y_true, y_pred
