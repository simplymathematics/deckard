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

import numpy as np
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

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


SCORE_PAYLOAD_SCHEMA = "deckard.score.v1"

SerializableScalar = str | int | float | bool | None
SerializableValue = (
    SerializableScalar
    | list["SerializableValue"]
    | dict[str, "SerializableValue"]
)


class ScoreDict(dict):
    """Dictionary-like score payload with canonical transformation helpers."""

    @staticmethod
    def normalize_value(value: Any) -> Any:
        """Convert runtime score values into JSON/YAML serializable values.

        Args:
            value: Runtime score payload value.

        Returns:
            JSON/YAML-safe representation of the provided value.
        """
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, ScoreDict):
            return {str(k): ScoreDict.normalize_value(v) for k, v in value.items()}
        if isinstance(value, dict):
            return {str(k): ScoreDict.normalize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ScoreDict.normalize_value(v) for v in value]
        if isinstance(value, np.ndarray):
            return ScoreDict.normalize_value(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Series):
            return {str(k): ScoreDict.normalize_value(v) for k, v in value.items()}
        if isinstance(value, pd.DataFrame):
            return {
                "columns": [str(c) for c in value.columns],
                "records": [
                    {str(k): ScoreDict.normalize_value(v) for k, v in row.items()}
                    for row in value.to_dict(orient="records")
                ],
            }
        return value

    @classmethod
    def from_payload(cls, payload: Any) -> "ScoreDict":
        """Build a ScoreDict from arbitrary score payload input.

        Args:
            payload: Arbitrary score payload to normalize.

        Returns:
            Normalized ScoreDict instance.
        """
        if isinstance(payload, ScoreDict):
            return cls(payload)
        if isinstance(payload, pd.Series):
            payload = payload.to_dict()
        if isinstance(payload, dict):
            return cls({str(k): cls.normalize_value(v) for k, v in payload.items()})
        return cls({"value": cls.normalize_value(payload)})

    def merge(self, other: dict[str, Any] | "ScoreDict") -> "ScoreDict":
        """Return merged score payload with `other` taking precedence.

        Args:
            other: Score payload to merge into the current mapping.

        Returns:
            Merged ScoreDict where other takes precedence.
        """
        merged = ScoreDict.from_payload(self)
        merged.update(ScoreDict.from_payload(other))
        return merged

    def update_score(
        self,
        value: Any,
        *,
        key: str | None = None,
        stage: str | None = None,
        mode: str | None = None,
        split: str | None = None,
    ) -> "ScoreDict":
        """Update score payload optionally nested by stage/mode/split.

        Args:
            value: Score value or nested score mapping.
            key: Optional key for the value at the resolved cursor.
            stage: Optional stage key.
            mode: Optional mode key.
            split: Optional split key.

        Returns:
            The updated ScoreDict instance.
        """
        cursor: dict[str, Any] = self
        for token in (stage, mode, split):
            if token is None:
                continue
            token_key = str(token)
            existing = cursor.get(token_key)
            if not isinstance(existing, dict):
                existing = {}
                cursor[token_key] = existing
            cursor = existing

        if key is None:
            if isinstance(value, dict):
                cursor.update(ScoreDict.from_payload(value))
            else:
                cursor["value"] = ScoreDict.normalize_value(value)
        else:
            cursor[str(key)] = ScoreDict.normalize_value(value)
        return self

    def get_scores(
        self,
        *,
        stage: str | None = None,
        mode: str | None = None,
        split: str | None = None,
        default: Any = None,
    ) -> Any:
        """Return score payload view optionally narrowed by stage/mode/split.

        Args:
            stage: Optional stage key.
            mode: Optional mode key.
            split: Optional split key.
            default: Fallback value when path resolution fails.

        Returns:
            Resolved score payload or default.
        """
        cursor: Any = self
        for token in (stage, mode, split):
            if token is None:
                continue
            if not isinstance(cursor, dict):
                return default
            cursor = cursor.get(str(token), default)
            if cursor is default:
                return default
        return cursor

    @staticmethod
    def _flatten(payload: Any, prefix: str = "", sep: str = ".") -> dict[str, Any]:
        flattened: dict[str, Any] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_text = str(key)
                next_prefix = key_text if prefix == "" else f"{prefix}{sep}{key_text}"
                flattened.update(ScoreDict._flatten(value, next_prefix, sep=sep))
            return flattened
        if isinstance(payload, list):
            key = prefix if prefix != "" else "value"
            flattened[key] = payload
            return flattened
        key = prefix if prefix != "" else "value"
        flattened[key] = payload
        return flattened

    def flatten(self, sep: str = ".") -> dict[str, Any]:
        """Return a dot-delimited flat score mapping.

        Args:
            sep: Token separator used for flattened keys.

        Returns:
            Flattened score mapping.
        """
        return ScoreDict._flatten(dict(self), sep=sep)

    def flat_by_scope(self, sep: str = ".") -> dict[str, dict[str, Any]]:
        """Group flattened score keys by first token scope.

        Args:
            sep: Token separator used for flattened keys.

        Returns:
            Mapping from scope token to scoped flat score mapping.
        """
        grouped: dict[str, dict[str, Any]] = {}
        for key, value in self.flatten(sep=sep).items():
            parts = str(key).split(sep, 1)
            scope = parts[0] if len(parts) > 0 and parts[0] != "" else "root"
            scoped_key = parts[1] if len(parts) == 2 else "value"
            grouped.setdefault(scope, {})[scoped_key] = value
        return grouped

    def dotlist_dict(self, sep: str = ".") -> dict[str, Any]:
        """Return OmegaConf-style dot-key dictionary.

        Args:
            sep: Token separator used for flattened keys.

        Returns:
            Dot-key dictionary for OmegaConf-style overrides.
        """
        return dict(self.flatten(sep=sep))

    def dotlist_items(self, sep: str = ".") -> list[str]:
        """Return OmegaConf-style `key=value` entries.

        Args:
            sep: Token separator used for flattened keys.

        Returns:
            List of key=value override strings.
        """
        return [
            f"{k}={json.dumps(ScoreDict.normalize_value(v), default=str)}"
            for k, v in self.flatten(sep=sep).items()
        ]

    def to_contract_envelope(self, schema: str = "deckard.score.v1") -> dict[str, Any]:
        """Return standardized serialization envelope for persisted scores.

        Args:
            schema: Envelope schema identifier.

        Returns:
            Standardized score envelope containing nested and flattened payloads.
        """
        flat = self.flatten(sep=".")
        return {
            "_schema": schema,
            "payload": dict(self),
            "flat": flat,
            "flat_by_scope": self.flat_by_scope(sep="."),
            "dotlist": dict(flat),
            "dotlist_items": self.dotlist_items(sep="."),
        }

    def __call__(
        self,
        *,
        score_file: str | None = None,
        artifact_loader: Any = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Load/merge/save lifecycle for score payloads.

        When ``score_file`` is provided and ``artifact_loader`` has
        ``load_scores``/``save_scores``, this call always performs a disk read
        (if the file exists) and a disk write (when ``persist=True``).
        Without ``score_file``, returns the in-memory nested payload.

        Args:
            score_file: Optional persisted score path.
            artifact_loader: Optional loader implementing load_scores/save_scores.
            persist: Whether to persist merged scores when score_file is provided.

        Returns:
            Resolved nested score payload as a dictionary.
        """
        current = ScoreDict.from_payload(self)
        if score_file is None:
            return dict(current)

        score_path = Path(score_file)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        merged = ScoreDict.from_payload(current)

        if score_path.exists() and artifact_loader is not None and hasattr(
            artifact_loader,
            "load_scores",
        ):
            disk_scores = artifact_loader.load_scores(score_file)
            merged = merged.merge(ScoreDict.from_payload(disk_scores))

        if persist and artifact_loader is not None and hasattr(
            artifact_loader,
            "save_scores",
        ):
            artifact_loader.save_scores(dict(merged), score_file)

        self.clear()
        self.update(merged)
        return dict(self)


def _normalize_score_value(value: Any) -> Any:
    """Convert runtime score payloads into JSON/YAML-serializable values."""
    return ScoreDict.normalize_value(value)


def _flatten_score_payload(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested score payload into dot-delimited key/value pairs."""
    score_dict = ScoreDict.from_payload(payload)
    if prefix == "":
        return score_dict.flatten(sep=".")
    return ScoreDict._flatten(score_dict, prefix=prefix, sep=".")


def _flat_by_scope(flat_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Group flat score keys by first dot-token runtime scope."""
    return ScoreDict.from_payload(flat_payload).flat_by_scope(sep=".")


def _serialize_scores_payload(scores: dict[str, Any] | pd.Series) -> dict[str, Any]:
    """Build canonical score payload envelope for persistence."""
    score_dict = ScoreDict.from_payload(scores)
    envelope = score_dict.to_contract_envelope(schema=SCORE_PAYLOAD_SCHEMA)
    # Backward compatibility: keep payload keys at top-level for readers that
    # still expect flat/nested score keys directly in JSON/YAML root.
    for key, value in dict(score_dict).items():
        if key not in envelope:
            envelope[key] = value
    return envelope


def _deserialize_scores_payload(raw: Any) -> dict[str, Any]:
    """Decode persisted score payloads into runtime-facing score dictionaries."""
    if not isinstance(raw, dict):
        return {"data": raw}
    if raw.get("_schema") == SCORE_PAYLOAD_SCHEMA:
        payload = raw.get("payload", {})
        return dict(payload) if isinstance(payload, dict) else {"data": payload}

    files = raw.pop("files", None)
    params = raw.pop("params", None)
    scores = dict(raw)
    if files is not None:
        scores["files"] = files
    if params is not None:
        scores["params"] = params
    return scores


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
        scores: ScoreDict | dict[str, SerializableValue] | pd.Series,
        filepath: Optional[str] = None,
    ) -> None:
        """Persist score payloads to disk in a supported tabular/structured format.

        Args:
            scores: Score payload mapping or series to persist.
            filepath: Target file path.

        Raises:
            ImportError: If YAML persistence is requested without PyYAML.
            ValueError: If filepath extension is unsupported.
            AssertionError: If filepath is missing or file was not persisted.
        """
        assert filepath is not None, "Filepath must be provided to save scores."
        score_path = Path(filepath)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        supported_filetypes = [".csv", ".json", ".xlsx", ".yaml", ".yml"]
        serialized = _serialize_scores_payload(scores)
        if score_path.suffix in supported_filetypes:
            match score_path.suffix:
                case ".csv":
                    csv_row = {
                        key: (
                            json.dumps(value)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for key, value in serialized["flat"].items()
                    }
                    pd.DataFrame([csv_row]).to_csv(score_path, index=False)
                case ".json":
                    with open(score_path, "w", encoding="utf-8") as f:
                        json.dump(serialized, f, indent=2)
                case ".xlsx":
                    xlsx_row = {
                        key: (
                            json.dumps(value)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        for key, value in serialized["flat"].items()
                    }
                    pd.DataFrame([xlsx_row]).to_excel(score_path, index=False)
                case ".yaml" | ".yml":
                    if yaml is None:
                        raise ImportError(
                            "PyYAML is required to save .yaml score artifacts",
                        )
                    with open(score_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(serialized, f, sort_keys=False)
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
            )
        assert Path(score_path).exists(), f"Failed to save scores to {score_path}"

    def load_scores(self, filepath: str) -> ScoreDict:
        """Load score payloads from disk and normalize key/value types.

        Args:
            filepath: Source score file path.

        Returns:
            Canonical score payload mapping.

        Raises:
            ImportError: If YAML loading is requested without PyYAML.
            ValueError: If filepath extension is unsupported.
            AssertionError: If score file does not exist.
        """
        score_path = Path(filepath)
        assert score_path.exists(), f"File {filepath} does not exist."
        supported_filetypes = [".csv", ".json", ".xlsx", ".yaml", ".yml"]
        scores: dict
        if score_path.suffix in supported_filetypes:
            match score_path.suffix:
                case ".csv":
                    df = pd.read_csv(score_path)
                    raw_scores = {} if len(df) == 0 else df.iloc[0].to_dict()
                    scores = {}
                    for key, value in raw_scores.items():
                        if isinstance(value, str):
                            text = value.strip()
                            if text.startswith("{") or text.startswith("["):
                                try:
                                    scores[str(key)] = json.loads(text)
                                    continue
                                except Exception:
                                    pass
                        scores[str(key)] = value
                case ".json":
                    with open(score_path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    scores = _deserialize_scores_payload(raw)
                case ".xlsx":
                    df = pd.read_excel(score_path)
                    raw_scores = {} if len(df) == 0 else df.iloc[0].to_dict()
                    scores = {}
                    for key, value in raw_scores.items():
                        if isinstance(value, str):
                            text = value.strip()
                            if text.startswith("{") or text.startswith("["):
                                try:
                                    scores[str(key)] = json.loads(text)
                                    continue
                                except Exception:
                                    pass
                        scores[str(key)] = value
                case ".yaml" | ".yml":
                    if yaml is None:
                        raise ImportError(
                            "PyYAML is required to load .yaml score artifacts",
                        )
                    with open(score_path, "r", encoding="utf-8") as f:
                        raw = yaml.safe_load(f)
                    scores = _deserialize_scores_payload(raw)
                case _:
                    raise ValueError(
                        f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
                    )
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
            )
        return ScoreDict.from_payload({str(k): v for k, v in scores.items()})

    def save_data(
        self,
        data: MatrixLike | ArrayLike | pd.DataFrame,
        filepath: Optional[str] = None,
        **kwargs: SerializableValue,
    ) -> None:
        """Persist matrix/array-like data payloads using file-suffix routing.

        Args:
            data: Matrix-like payload to persist.
            filepath: Target file path.
            **kwargs: Backend-specific pandas writer options.

        Raises:
            ValueError: If filepath extension is unsupported.
            AssertionError: If filepath is missing or file was not persisted.
        """
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

    def load_data(
        self,
        filepath: str,
        **kwargs: SerializableValue,
    ) -> MatrixLike | ArrayLike:
        """Load matrix/array-like payloads from disk based on file extension.

        Args:
            filepath: Source payload path.
            **kwargs: Backend-specific pandas reader options.

        Returns:
            Loaded matrix-like or array-like payload.

        Raises:
            FileNotFoundError: If filepath is None.
            ValueError: If filepath extension is unsupported.
        """
        if filepath is None:
            raise FileNotFoundError("Filepath is None.")

        path = Path(filepath)
        supported_filetypes = [
            ".csv",
            ".json",
            ".xlsx",
            ".parquet",
            ".pkl",
            ".npz",
            ".html",
        ]

        match path.suffix:
            case ".pkl":
                data = pd.read_pickle(path, **kwargs)
            case ".csv":
                data = pd.read_csv(path, **kwargs)
            case ".json":
                json_kwargs = {"orient": "records", **kwargs}
                if "lines" not in json_kwargs:
                    try:
                        data = pd.read_json(path, lines=True, **json_kwargs)
                    except ValueError:
                        data = pd.read_json(path, **json_kwargs)
                else:
                    data = pd.read_json(path, **json_kwargs)
            case ".xlsx":
                data = pd.read_excel(path, **kwargs)
            case ".parquet":
                data = pd.read_parquet(path, **kwargs)
            case ".html":
                data = pd.read_html(path, **kwargs)[0]
            case ".npz":
                npz_payload = np.load(path, allow_pickle=True)
                if len(npz_payload.files) == 0:
                    data = pd.DataFrame()
                else:
                    first_key = npz_payload.files[0]
                    data = pd.DataFrame(npz_payload[first_key])
            case _:
                raise ValueError(
                    f"Unsupported file type {path.suffix}. Supported types: {supported_filetypes}",
                )
        return data

    def load_matrix(
        self,
        filepath: str,
        **kwargs: SerializableValue,
    ) -> MatrixLike:
        """Load and return a matrix-like payload from disk.

        Args:
            filepath: Source payload path.
            **kwargs: Backend-specific pandas reader options.

        Returns:
            Loaded matrix-like payload.
        """
        return self.load_data(filepath, **kwargs)

    def load_vector(
        self,
        filepath: str,
        **kwargs: SerializableValue,
    ) -> ArrayLike:
        """Load and return a vector-like payload from disk.

        Args:
            filepath: Source payload path.
            **kwargs: Backend-specific pandas reader options.

        Returns:
            Loaded vector-like payload.
        """
        return self.load_data(filepath, **kwargs)

    def save_object(self, obj: EstimatorLike | SerializableValue, filepath: str) -> None:
        """Serialize a Python object payload with pickle-compatible formats.

        Args:
            obj: Object payload to serialize.
            filepath: Target pickle path.

        Raises:
            ValueError: If filepath extension is not pickle-compatible.
        """
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
    ) -> EstimatorLike | SerializableValue | None:
        """Load a pickled object payload with optional corrupt-file handling.

        Args:
            filepath: Source pickle path.
            ignore_corrupt: Return ``None`` when file is corrupt.
            delete_corrupt: Delete corrupt file before returning/raising.

        Returns:
            Loaded payload or ``None`` when ``ignore_corrupt`` is enabled.

        Raises:
            EOFError: If the pickle stream is truncated and ignore_corrupt is False.
            pickle.UnpicklingError: If pickle data is invalid and ignore_corrupt is False.
            AttributeError: If object attributes cannot be restored and ignore_corrupt is False.
            OSError: If file IO fails and ignore_corrupt is False.
        """
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
    ) -> EstimatorLike | SerializableValue | None:
        """Load model artifacts from .pt/.joblib/pickle-backed formats.

        Args:
            filepath: Source model artifact path.
            ignore_corrupt: Return ``None`` when file is corrupt.
            delete_corrupt: Delete corrupt file before returning/raising.

        Returns:
            Loaded model payload.

        Raises:
            ImportError: If required optional loaders (torch/joblib) are unavailable.
        """
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

    def save_model(self, model: EstimatorLike | SerializableValue, filepath: str) -> None:
        """Persist model artifacts using suffix-driven serialization backends.

        Args:
            model: Model payload to persist.
            filepath: Target model artifact path.

        Raises:
            ImportError: If required optional savers (torch/joblib) are unavailable.
            NotImplementedError: If model file extension is unsupported.
        """
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

    def save(
        self,
        payload: EstimatorLike | MatrixLike | ArrayLike | ScoreDict | SerializableValue | None = None,
        filepath: Optional[str] = None,
    ) -> None:
        """Persist payloads by delegating to score/data/object/model save handlers.

        Args:
            payload: Payload to persist; defaults to config metadata when omitted.
            filepath: Target artifact path.

        Raises:
            ValueError: If filepath is missing or extension is unsupported.
        """
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

    def load(
        self,
        filepath: Optional[str] = None,
    ) -> "ArtifactLoaderConfig | ScoreDict | MatrixLike | ArrayLike | EstimatorLike | SerializableValue | None":
        """Load artifacts from disk and dispatch to the appropriate loader.

        Args:
            filepath: Optional source path; defaults to ``self.path``.

        Returns:
            Loaded payload or the config instance when loading metadata envelope.

        Raises:
            ValueError: If path extension is unsupported for configured payload kind.
        """
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
