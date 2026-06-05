"""Canonical file-schema helpers for runtime path templates and placeholder resolution.

This module owns the file-key TypedDict registry, validation, placeholder parsing,
and job-specific path substitution used by runtime configs.
It does not own generic persistence or scoring serialization.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Mapping
from typing import Any, TypedDict
from uuid import uuid4

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .attack.canon import AttackFiles
from .data.canon import BaseFiles, DataFiles
from .detector.canon import DetectorFiles
from .model.canon import DefenseFiles, ModelFiles
from .utils import load_class
from .path_utils import to_posix_path


class LogFiles(TypedDict, total=False):
    """Typed mapping for log and error file paths.

    Attributes:
        log_file: Runtime log artifact path.
        error_file: Runtime error log artifact path.
    """

    log_file: str
    error_file: str


# -----------------------------------------------------------------------------
# key registry
# -----------------------------------------------------------------------------


def collect_typed_dict_keys(*td_classes: type[Any]) -> set[str]:
    keys: set[str] = set()
    for cls in td_classes:
        keys |= set(cls.__annotations__.keys())
    return keys


_ALLOWED_KEYS = collect_typed_dict_keys(
    ModelFiles,
    AttackFiles,
    LogFiles,
    BaseFiles,
    DataFiles,
    DefenseFiles,
    DetectorFiles,
)

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

# Key registries for each file type category
data_files = tuple(collect_typed_dict_keys(BaseFiles, DataFiles, LogFiles))
model_files = tuple(collect_typed_dict_keys(ModelFiles))
attack_files = tuple(collect_typed_dict_keys(AttackFiles))


class FileConfigError(TypeError):
    """Raised when file-key payloads violate the canonical file schema."""

    pass


class AbstractFileHandler(ABC):
    """Abstract file handler for canonical file-schema operations."""

    @abstractmethod
    def validate_keys(
        self,
        keys: Mapping[str, Any] | list[str] | tuple[str, ...],
    ) -> None:
        """Validate provided file keys against the allowed file schema.

        Args:
            keys: File key container to validate against allowed schema keys.
        """

    @abstractmethod
    def disk_status(self, files: Mapping[str, Any]) -> dict[str, bool]:
        """Return exists/not-exists status for provided file path mapping.

        Args:
            files: Mapping of file-key to filepath values.

        Returns:
            Mapping of file-key to disk existence status.
        """

    @abstractmethod
    def parse_placeholders(self, value: str) -> list[str]:
        """Parse placeholder tokens from a template string.

        Args:
            value: Template string containing placeholder markers.

        Returns:
            Placeholder token list extracted from value.
        """

    @abstractmethod
    def replace_placeholders(self, value: str, replacements: Mapping[str, Any]) -> str:
        """Apply placeholder replacements to a template string.

        Args:
            value: Template string containing placeholder markers.
            replacements: Mapping of placeholder tokens to replacement values.

        Returns:
            Resolved string with placeholders replaced.
        """


class CanonFileHandler(AbstractFileHandler):
    """Concrete handler operating on canonical file-schema key/value payloads."""

    _placeholder_re = re.compile(r"\{[^{}]+\}")

    def validate_keys(
        self,
        keys: Mapping[str, Any] | list[str] | tuple[str, ...],
    ) -> None:
        """Validate incoming key set against the canonical key registry.

        Args:
            keys: File key container to validate.

        Raises:
            FileConfigError: If one or more keys are not in the canonical schema.
        """
        iterable = keys.keys() if isinstance(keys, Mapping) else keys
        invalid = [key for key in iterable if key not in _ALLOWED_KEYS]
        if invalid:
            raise FileConfigError(f"Invalid file key(s): {', '.join(sorted(invalid))}")

    def disk_status(self, files: Mapping[str, Any]) -> dict[str, bool]:
        """Return per-key disk existence status for provided file mapping.

        Args:
            files: Mapping of file-key to filepath values.

        Returns:
            Mapping of file-key to disk existence status.
        """
        self.validate_keys(files)
        status: dict[str, bool] = {}
        for key, value in files.items():
            if isinstance(value, str) and value.strip() != "":
                status[key] = Path(value).exists()
            else:
                status[key] = False
        return status

    def parse_placeholders(self, value: str) -> list[str]:
        """Parse placeholder tokens from a template string.

        Args:
            value: Template string containing placeholder markers.

        Returns:
            Placeholder token list extracted from value.
        """
        return self._placeholder_re.findall(value)

    def replace_placeholders(self, value: str, replacements: Mapping[str, Any]) -> str:
        """Apply placeholder replacements to a template string.

        Args:
            value: Template string containing placeholder markers.
            replacements: Mapping of placeholder tokens to replacement values.

        Returns:
            Resolved template string.
        """
        resolved = value
        for token, replacement in replacements.items():
            resolved = resolved.replace(str(token), str(replacement))
        return resolved

    def to_dict(self):
        return {"_target_": "deckard.file.CanonFileHandler"}


# -----------------------------------------------------------------------------
# resolver
# -----------------------------------------------------------------------------


class PlaceholderResolverMixin:
    """Reusable placeholder-expansion helpers for file-oriented config objects.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    replace: dict[str, str]

    @property
    def num(self) -> str:
        """Return the serial job number in a multirun sweep.

        Returns:
            Hydra job number when available, otherwise a UUID fallback.
        """
        if hasattr(self, "_num_override"):
            return str(self._num_override)
        try:
            return str(HydraConfig.get().job.num)
        except Exception:
            return uuid4().hex

    @num.setter
    def num(self, value: int) -> None:
        """Set the resolved job num override.

        Args:
            value: Job number override.
        """
        self._num_override = str(value)

    @property
    def id(self) -> str:
        """Return the specific launcher or cluster job identifier.

        Returns:
            Hydra job ID/num when available, otherwise a UUID fallback.
        """
        if hasattr(self, "_id_override"):
            return str(self._id_override)
        try:
            return str(HydraConfig.get().job.id)
        except Exception:
            try:
                return str(HydraConfig.get().job.num)
            except Exception:
                return uuid4().hex

    @id.setter
    def id(self, value: int) -> None:
        """Set the resolved job id override.

        Args:
            value: Job ID override.
        """
        self._id_override = str(value)

    def _replacement_dict(self) -> dict[str, str]:
        replacements = {
            "{num}": self.num,
            "{#}": self.num,
            "{timestamp}": time.strftime("%Y%m%d-%H%M%S"),
            "{hash}": self.id,
            "{*}": self.id,
        }
        replacements.update(
            {k: str(v) for k, v in getattr(self, "replace", {}).items()},
        )
        return replacements

    def _resolve(self, value: str | None) -> str | None:
        if not value:
            return None
        handler = getattr(self, "handler", None)
        replacements = self._replacement_dict()
        if isinstance(handler, AbstractFileHandler):
            return handler.replace_placeholders(value, replacements)
        resolved = value
        for token, replacement in replacements.items():
            resolved = resolved.replace(token, replacement)
        return resolved


class FileConfig(PlaceholderResolverMixin):
    """Dynamic file-path configuration container.

    ``FileConfig`` stores validated artifact paths for datasets, models,
    predictions, logs, attacks, and scores.

    Placeholder expansion is applied to string values. Supported placeholders:
    - ``{num}``: Hydra job number.
    - ``{timestamp}``: Current timestamp (``YYYYMMDD-HHMMSS``).
    - ``{hash}``: Hash/job id for the file config.
    - ``{#}`` and ``{*}``: Alias placeholders.

    Args:
        replace: Optional mapping used for additional placeholder replacements.
        handler: Optional shared file handler implementation.
        **files: File-path keyword arguments matching the configured schema.

    Raises:
        FileConfigError: If an unknown file key is provided.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def __init__(
        self,
        *,
        replace: dict[str, str] | None = None,
        handler: AbstractFileHandler | str | None = None,
        **files: Any,
    ):
        self.replace = replace or {}
        if isinstance(handler, str):
            self.handler = load_class(str)
        elif isinstance(handler, dict):
            self.handler = load_class(**handler)
        elif isinstance(handler, DictConfig):
            self.handler = instantiate(handler)
        else:
            self.handler = handler or CanonFileHandler()
        self._files: dict[str, Any] = {}
        self._raw_files: dict[str, Any] = {}

        self.handler.validate_keys(files)
        for k, v in files.items():
            self._set(k, v)

    @classmethod
    def from_payload(cls, payload: Any) -> "FileConfig":
        """Coerce a dict-like files payload into a FileConfig instance.

        Args:
            payload: Files payload mapping, DictConfig, or FileConfig.

        Returns:
            Normalized FileConfig instance.

        Raises:
            TypeError: If payload is not mapping-like.
        """
        if isinstance(payload, cls):
            return payload
        if payload is None:
            return cls()
        if isinstance(payload, DictConfig):
            payload = OmegaConf.to_container(payload, resolve=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"files payload must be mapping-like. Got {type(payload)}")

        replace = payload.get("replace")
        handler = payload.get("handler")
        file_values = {
            key: value for key, value in payload.items() if key in _ALLOWED_KEYS
        }
        return cls(
            replace=dict(replace) if isinstance(replace, Mapping) else None,
            handler=handler if isinstance(handler, AbstractFileHandler) else None,
            **file_values,
        )

    # -------------------------------------------------------------------------
    # validation
    # -------------------------------------------------------------------------

    def _validate_key(self, key: str) -> None:
        self.handler.validate_keys([key])

    # -------------------------------------------------------------------------
    # assignment
    # -------------------------------------------------------------------------

    def _set(self, key: str, value: Any) -> None:
        raw_value = value
        if isinstance(value, str):
            value = self._resolve(value)

        self._raw_files[key] = raw_value
        self._files[key] = value
        setattr(self, key, value)

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------

    def update(self, **kwargs: Any) -> None:
        """Validate and assign file-path fields on the runtime file mapping.

        Args:
            **kwargs: File mapping entries to set/update.
        """
        for k, v in kwargs.items():
            self._validate_key(k)
            self._set(k, v)

    def apply_runtime_paths(self, **kwargs: Any) -> None:
        """Assign resolved runtime paths for configured file fields.

        Args:
            **kwargs: Runtime file field updates keyed by file field name.
        """
        runtime_updates = {
            key: value for key, value in kwargs.items() if value is not None
        }
        for key, value in runtime_updates.items():
            self._validate_key(key)
            self._files[key] = value
            setattr(self, key, value)

    def hash_artifact_paths(
        self,
        context_hash: str,
        *,
        exclude: Collection[str] = (),
    ) -> None:
        """Hash runtime artifact filenames while preserving init templates.

        Args:
            context_hash: Stable hash used for artifact basename rewriting.
            exclude: File keys excluded from path hashing.
        """
        for key, value in list(self._files.items()):
            if (
                key.endswith("_file")
                and key not in exclude
                and isinstance(value, str)
                and value.strip() != ""
            ):
                hashed_value = self._hash_file_path_basename(value, context_hash)
                self._files[key] = hashed_value
                setattr(self, key, hashed_value)

    @staticmethod
    def _hash_file_path_basename(path_value: str, context_hash: str) -> str:
        path = Path(path_value)
        suffix = "".join(path.suffixes)
        hashed_name = f"{context_hash}{suffix}" if suffix else context_hash
        return to_posix_path(path.parent / hashed_name)

    def as_dict(self) -> dict[str, JsonValue]:
        """Return file mapping as a plain dictionary.

        Returns:
            Serialized file mapping dictionary.
        """
        return dict(self._files)

    def to_init_dict(self) -> dict[str, JsonValue]:
        """Return reproducible initialization payload without live helper objects.

        Returns:
            Constructor-safe file payload map.
        """
        payload = dict(self._raw_files)
        if self.replace:
            payload["replace"] = dict(self.replace)
        return payload

    def to_runtime_dict(self) -> dict[str, JsonValue]:
        """Return runtime-resolved file mapping for execution/config propagation.

        Returns:
            Runtime file payload map.
        """
        payload = self.as_dict()
        if self.replace:
            payload["replace"] = dict(self.replace)
        return payload

    def disk_status(self) -> dict[str, bool]:
        """Return per-file existence status for configured file paths.

        Returns:
            Mapping from configured file key to disk existence status.
        """
        return self.handler.disk_status(self._files)

    def __getitem__(self, key: str) -> Any:
        return self._files[key]

    def __iter__(self):
        return iter(self._files)

    def __len__(self) -> int:
        return len(self._files)
