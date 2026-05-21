"""Compatibility helpers for artifact loading.

This module provides the small public surface expected by integration tests
and older callers that imported ``deckard.artifacts`` directly.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd

from .frameworks.types import ArrayLike, EstimatorLike, MatrixLike

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None


@dataclass(eq=False, kw_only=True)
class ArtifactLoaderConfig:
    """Base artifact loader for file-backed deckard configs."""

    id: str = ""
    path: str = ""
    payload_kind: str = "data"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args, **kwds):
        self.args = args if args else ()

        dataclass_fields = self.__dataclass_fields__
        init_fields = [
            field_name
            for field_name, dataclass_field in dataclass_fields.items()
            if dataclass_field.init
        ]

        if len(args) > len(init_fields):
            raise TypeError(
                f"Expected at most {len(init_fields)} positional arguments, got {len(args)}",
            )

        for field_name, dataclass_field in dataclass_fields.items():
            if dataclass_field.default is not MISSING:
                setattr(self, field_name, dataclass_field.default)
            elif dataclass_field.default_factory is not MISSING:
                setattr(self, field_name, dataclass_field.default_factory())

        for index, arg in enumerate(args):
            setattr(self, init_fields[index], arg)
        for key, value in kwds.items():
            setattr(self, key, value)

        self._before_post_init()
        self.__post_init__()
        self._after_post_init()

    def __post_init__(self):
        pass

    def _before_post_init(self) -> None:
        """Hook for subclasses that need pre-normalization."""

    def _after_post_init(self) -> None:
        """Hook for subclasses that finalize derived state after init."""

    def save_scores(
        self,
        scores: dict[str, Any] | pd.Series,
        filepath: Optional[str] = None,
    ) -> None:
        assert filepath is not None, "Filepath must be provided to save scores."
        score_path = Path(filepath)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        supported_filetypes = [".csv", ".json", ".xlsx"]
        if not isinstance(scores, dict):
            scores = dict(scores)
        if score_path.suffix in supported_filetypes:
            match score_path.suffix:
                case ".csv":
                    pd.DataFrame([scores]).to_csv(score_path, index=False)
                case ".json":
                    with open(score_path, "w", encoding="utf-8") as f:
                        json.dump(scores, f, indent=4)
                case ".xlsx":
                    pd.DataFrame([scores]).to_excel(score_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
            )
        assert Path(score_path).exists(), f"Failed to save scores to {score_path}"

    def load_scores(self, filepath: str) -> dict[str, Any]:
        score_path = Path(filepath)
        assert score_path.exists(), f"File {filepath} does not exist."
        supported_filetypes = [".csv", ".json", ".xlsx"]
        scores: dict
        if score_path.suffix in supported_filetypes:
            match score_path.suffix:
                case ".csv":
                    df = pd.read_csv(score_path)
                    scores = {} if len(df) == 0 else df.iloc[0].to_dict()
                case ".json":
                    with open(score_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    if not isinstance(raw, dict):
                        raw = {"data": raw}
                    files = raw.pop("files", None)
                    params = raw.pop("params", None)
                    scores = raw
                    if files is not None:
                        scores["files"] = files
                    if params is not None:
                        scores["params"] = params
                case ".xlsx":
                    df = pd.read_excel(score_path)
                    scores = {} if len(df) == 0 else df.iloc[0].to_dict()
                case _:
                    raise ValueError(
                        f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
                    )
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
            )
        return {str(k): v for k, v in scores.items()}

    def save_data(
        self,
        data: MatrixLike | ArrayLike | pd.DataFrame,
        filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        assert filepath is not None, "Filepath must be provided to save data."
        data_path = Path(filepath)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        filetype = data_path.suffix
        supported_filetypes = [
            ".csv",
            ".parquet",
            ".pkl",
            ".html",
            ".json",
            ".xlsx",
        ]
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(cast(Any, data))
        match filetype:
            case ".pkl":
                data.to_pickle(data_path, **kwargs)
            case ".csv":
                data.to_csv(data_path, index=False, **kwargs)
            case ".parquet":
                data.to_parquet(data_path, index=False, **kwargs)
            case ".html":
                data.to_html(data_path, index=False, **kwargs)
            case ".json":
                data.to_json(data_path, orient="records", lines=True, **kwargs)
            case ".xlsx":
                data.to_excel(data_path, index=False, **kwargs)
            case _:
                raise ValueError(
                    f"Unsupported file type {data_path.suffix}. Supported types: {supported_filetypes}",
                )
        assert Path(data_path).exists(), f"Failed to save data to {data_path}"

    def load_data(self, filepath: str, **kwargs: Any) -> MatrixLike | ArrayLike:
        from .utils import load_data as load_tabular_data

        return load_tabular_data(filepath, **kwargs)

    def load_matrix(
        self,
        filepath: str,
        **kwargs: Any,
    ) -> MatrixLike:
        return self.load_data(filepath, **kwargs)

    def load_vector(
        self,
        filepath: str,
        **kwargs: Any,
    ) -> ArrayLike:
        return self.load_data(filepath, **kwargs)

    def save_object(self, obj: EstimatorLike | Any, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        suffix = Path(filepath).suffix
        supported_suffixes = [".pkl", ".pickle"]
        if suffix not in supported_suffixes:
            raise ValueError(
                f"Unsupported file type {suffix}. Supported types: {supported_suffixes}",
            )
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

    def load_object(
        self,
        filepath: str,
        ignore_corrupt: bool = False,
        delete_corrupt: bool = False,
    ) -> Any:
        try:
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, AttributeError, OSError):
            if delete_corrupt:
                Path(filepath).unlink(missing_ok=True)
            if ignore_corrupt:
                return None
            raise
        return obj

    def load_model(
        self,
        filepath: str,
        ignore_corrupt: bool = False,
        delete_corrupt: bool = False,
    ) -> EstimatorLike | Any:
        suffix = Path(filepath).suffix.lower()
        if suffix == ".pt":
            if torch is None:
                raise ImportError(
                    "torch is required to load .pt model artifacts",
                )
            return torch.load(filepath, map_location="cpu")
        if suffix == ".joblib":
            if joblib is None:
                raise ImportError(
                    "joblib is required to load .joblib model artifacts",
                )
            return joblib.load(filepath)
        return self.load_object(
            filepath,
            ignore_corrupt=ignore_corrupt,
            delete_corrupt=delete_corrupt,
        )

    def save_model(self, model: EstimatorLike | Any, filepath: str) -> None:
        suffix = Path(filepath).suffix.lower()
        if suffix == ".pt":
            if torch is None:
                raise ImportError(
                    "torch is required to save .pt model artifacts",
                )
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model, filepath)
            return
        elif suffix == ".joblib":
            if joblib is None:
                raise ImportError(
                    "joblib is required to save .joblib model artifacts",
                )
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, filepath)
            return
        elif suffix in [".pkl", ".pickle"]:
            self.save_object(model, filepath)
        else:
            raise NotImplementedError(
                f"Saving models with extension: {suffix} not supported.",
            )

    def save(self, payload: Any = None, filepath: Optional[str] = None) -> None:
        # Backward compatibility: many callers use save(filepath) positional style.
        if filepath is None and isinstance(payload, (str, Path)):
            filepath = str(payload)
            payload = self

        path = Path(filepath or self.path)
        if not str(path):
            raise ValueError("Filepath must be provided to save artifacts.")

        suffix = path.suffix.lower()
        if payload is None:
            payload = {
                "id": self.id,
                "payload_kind": self.payload_kind,
                "metadata": self.metadata,
            }

        if (
            suffix == ".json"
            and isinstance(payload, dict)
            and {
                "id",
                "payload_kind",
                "metadata",
            }.intersection(payload)
        ):
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=4)
            return

        if suffix in {".pkl", ".pickle"}:
            if self.payload_kind in {"model", "estimator"}:
                self.save_model(payload, str(path))
                return
            self.save_object(payload, str(path))
            return
        if suffix == ".pt":
            self.save_model(payload, str(path))
            return
        if self.payload_kind in {"score", "scores"}:
            self.save_scores(payload, str(path))
            return
        if suffix in {".csv", ".parquet", ".html", ".json", ".xlsx"}:
            self.save_data(payload, str(path))
            return

        raise ValueError(f"Unsupported file type {path.suffix}.")

    def load(self, filepath: Optional[str] = None) -> Any:
        path = Path(filepath or self.path)
        if not path.exists():
            return self

        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                payload = None

            if isinstance(payload, dict) and {
                "id",
                "payload_kind",
                "metadata",
            }.intersection(payload):
                self.id = str(payload.get("id", self.id))
                self.payload_kind = str(payload.get("payload_kind", self.payload_kind))
                metadata = payload.get("metadata", None)
                if isinstance(metadata, dict):
                    self.metadata = metadata
                return self

            if self.payload_kind in {"score", "scores"}:
                return self.load_scores(str(path))
            return self.load_data(str(path))

        if suffix in {".csv", ".xlsx"}:
            if self.payload_kind in {"score", "scores"}:
                return self.load_scores(str(path))
            return self.load_data(str(path))

        if suffix in {".parquet", ".html"}:
            return self.load_data(str(path))

        if suffix in {".pkl", ".pickle"}:
            if self.payload_kind in {"model", "estimator"}:
                return self.load_model(str(path))
            return self.load_object(str(path))

        if suffix == ".pt":
            return self.load_model(str(path))

        raise ValueError(f"Unsupported file type {path.suffix}.")
