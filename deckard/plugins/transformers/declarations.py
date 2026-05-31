"""Explicit Hugging Face dataset declarations for transformers workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from ...data.base import DataConfig
from ...utils import coerce_to_list

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None


@dataclass(eq=False, kw_only=True)
class FlexibleHuggingFaceDataset(DataConfig):
    """DataConfig specialization for explicit Hugging Face dataset loading.

    The caller must provide the dataset identity, exact split token, target
    column, and feature columns. This keeps dataset construction explicit and
    avoids split normalization or column inference.
    """

    name: str
    target: str
    keep: list[str]
    dataset_split: str
    dataset_config_name: str | None = None
    limit: int | None = None
    data_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.keep = list(coerce_to_list(self.keep))
        if len(self.keep) == 0:
            raise ValueError("keep must include at least one explicit feature column")
        if str(self.dataset_split).strip() == "":
            raise ValueError("dataset_split must be provided")
        if str(self.target).strip() == "":
            raise ValueError("target must be provided")

        super().__post_init__()
        self._target_ = (
            "deckard.plugins.transformers.declarations.FlexibleHuggingFaceDataset"
        )

    def _load_huggingface_dataset(self) -> None:
        if load_dataset is None:
            raise ImportError(
                "FlexibleHuggingFaceDataset requires 'datasets'. Install deckard[datasets].",
            )

        loader_kwargs = dict(self.data_params or {})
        if self.dataset_config_name is None:
            dataset = load_dataset(
                self.name,
                split=self.dataset_split,
                **loader_kwargs,
            )
        else:
            dataset = load_dataset(
                self.name,
                self.dataset_config_name,
                split=self.dataset_split,
                **loader_kwargs,
            )

        if hasattr(dataset, "to_pandas"):
            frame = dataset.to_pandas()
        else:
            frame = pd.DataFrame(dataset)

        if self.limit is not None:
            frame = frame.head(int(self.limit))

        missing_columns = [
            column
            for column in [*self.keep, self.target]
            if column not in frame.columns
        ]
        if missing_columns:
            raise KeyError(
                "FlexibleHuggingFaceDataset missing required columns: "
                f"{sorted(missing_columns)}",
            )

        self._X = frame.loc[:, self.keep].copy()
        self._y = frame.loc[:, self.target].copy()

        if not isinstance(self._y, pd.Series):
            self._y = pd.Series(self._y, name=self.target)

    def load_dataset(self) -> "FlexibleHuggingFaceDataset":
        """Load the configured Hugging Face dataset into ``_X`` and ``_y``.

        The method requires the optional ``datasets`` dependency, forwards
        ``name``, ``dataset_config_name``, ``dataset_split``, and ``data_params``
        to :func:`datasets.load_dataset`, truncates the frame with ``limit``
        before validating columns, and then materializes ``_X`` from ``keep``
        and ``_y`` from ``target``. Missing required columns raise ``KeyError``.

        Returns:
            The current dataset config after loading runtime state.
        """
        start_time = time.process_time()
        self._load_dataset_with_hooks(self._load_huggingface_dataset)
        end_time = time.process_time()
        self._set_time("data_load_time", end_time - start_time)
        self.target = str(self.target)
        self.keep = list(self.keep)
        return cast("FlexibleHuggingFaceDataset", self)


__all__ = ["FlexibleHuggingFaceDataset"]
