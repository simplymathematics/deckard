from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping
from typing import Any, TypedDict
from uuid import uuid4

from hydra.core.hydra_config import HydraConfig


class ModelFiles(TypedDict, total=False):
    model_file: str
    training_predictions_file: str
    test_predictions_file: str
    training_probabilities_file: str
    test_probabilities_file: str
    score_file: str


class AttackFiles(TypedDict, total=False):
    attack_file: str
    attack_predictions_file: str
    score_file: str


class LogFiles(TypedDict, total=False):
    log_file: str
    error_file: str


class BaseFiles(TypedDict, total=False):
    data_file: str
    params_file: str
    score_file: str


class DefenseFiles(TypedDict, total=False):
    defended_model_file: str
    defended_predictions_file: str
    defended_probabilities_file: str
    score_file: str


class DetectorFiles(TypedDict, total=False):
    detector_model_file: str
    detected_predictions_file: str
    detected_probabilities_file: str
    score_file: str


# -----------------------------------------------------------------------------
# key registry
# -----------------------------------------------------------------------------


def collect_typed_dict_keys(*td_classes: type[TypedDict]) -> set[str]:
    keys: set[str] = set()
    for cls in td_classes:
        keys |= set(cls.__annotations__.keys())
    return keys


_ALLOWED_KEYS = collect_typed_dict_keys(
    ModelFiles,
    AttackFiles,
    LogFiles,
    BaseFiles,
    DefenseFiles,
    DetectorFiles,
)

# Key registries for each file type category
data_files = tuple(collect_typed_dict_keys(BaseFiles, LogFiles))
model_files = tuple(collect_typed_dict_keys(ModelFiles))
attack_files = tuple(collect_typed_dict_keys(AttackFiles))



class FileConfigError(TypeError):
    pass


class AbstractFileHandler(ABC):
    """Abstract file handler for canonical file-schema operations."""

    @abstractmethod
    def validate_keys(self, keys: Mapping[str, Any] | list[str] | tuple[str, ...]) -> None:
        """Validate provided file keys against the allowed file schema."""

    @abstractmethod
    def disk_status(self, files: Mapping[str, Any]) -> dict[str, bool]:
        """Return exists/not-exists status for provided file path mapping."""

    @abstractmethod
    def parse_placeholders(self, value: str) -> list[str]:
        """Parse placeholder tokens from a template string."""

    @abstractmethod
    def replace_placeholders(self, value: str, replacements: Mapping[str, Any]) -> str:
        """Apply placeholder replacements to a template string."""


class CanonFileHandler(AbstractFileHandler):
    """Concrete handler operating on canonical file-schema key/value payloads."""

    _placeholder_re = re.compile(r"\{[^{}]+\}")

    def validate_keys(self, keys: Mapping[str, Any] | list[str] | tuple[str, ...]) -> None:
        iterable = keys.keys() if isinstance(keys, Mapping) else keys
        invalid = [key for key in iterable if key not in _ALLOWED_KEYS]
        if invalid:
            raise FileConfigError(f"Invalid file key(s): {', '.join(sorted(invalid))}")

    def disk_status(self, files: Mapping[str, Any]) -> dict[str, bool]:
        self.validate_keys(files)
        status: dict[str, bool] = {}
        for key, value in files.items():
            if isinstance(value, str) and value.strip() != "":
                status[key] = Path(value).exists()
            else:
                status[key] = False
        return status

    def parse_placeholders(self, value: str) -> list[str]:
        return self._placeholder_re.findall(value)

    def replace_placeholders(self, value: str, replacements: Mapping[str, Any]) -> str:
        resolved = value
        for token, replacement in replacements.items():
            resolved = resolved.replace(str(token), str(replacement))
        return resolved


# -----------------------------------------------------------------------------
# resolver
# -----------------------------------------------------------------------------


class PlaceholderResolverMixin:
    """Reusable placeholder-expansion helpers for file-oriented config objects."""

    replace: dict[str, str]

    @property
    def num(self) -> str:
        """Returns the serial job number in a multirun sweep. Uses uuid as fallback if Hydra is not enabled."""
        if hasattr(self, "_num_override"):
            return str(self._num_override)
        try:
            return str(HydraConfig.get().job.num)
        except Exception:
            return uuid4().hex

    @num.setter
    def num(self, value: int) -> None:
        """Set the resolved job num override."""
        self._num_override = str(value)

    @property
    def id(self) -> str:
        """Returns the specific launcher or cluster job ID. Uses uuid as fallback if Hydra is not enabled"""
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
        """Set the resolved job id override."""
        self._id_override = str(value)

    def _replacement_dict(self) -> dict[str, str]:
        replacements = {
            "{num}": self.num,
            "{#}": self.num,
            "{timestamp}": time.strftime("%Y%m%d-%H%M%S"),
            "{hash}": self.id,
            "{*}": self.id,
        }
        replacements.update({k: str(v) for k, v in getattr(self, "replace", {}).items()})
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
    """

    def __init__(
        self,
        *,
        replace: dict[str, str] | None = None,
        handler: AbstractFileHandler | None = None,
        **files: Any,
    ):
        self.replace = replace or {}
        self.handler = handler or CanonFileHandler()
        self._files: dict[str, Any] = {}

        self.handler.validate_keys(files)
        for k, v in files.items():
            self._set(k, v)

    # -------------------------------------------------------------------------
    # validation
    # -------------------------------------------------------------------------

    def _validate_key(self, key: str) -> None:
        self.handler.validate_keys([key])

    # -------------------------------------------------------------------------
    # assignment
    # -------------------------------------------------------------------------

    def _set(self, key: str, value: Any) -> None:
        if isinstance(value, str):
            value = self._resolve(value)

        self._files[key] = value
        setattr(self, key, value)

    # -------------------------------------------------------------------------
    # public API
    # -------------------------------------------------------------------------

    def update(self, **kwargs: Any) -> None:
        """Validate and assign file-path fields on the runtime file mapping."""
        for k, v in kwargs.items():
            self._validate_key(k)
            self._set(k, v)

    def as_dict(self) -> dict[str, Any]:
        """Return file mapping as a plain dictionary."""
        return dict(self._files)

    def disk_status(self) -> dict[str, bool]:
        """Return per-file existence status for configured file paths."""
        return self.handler.disk_status(self._files)

    def __getitem__(self, key: str) -> Any:
        return self._files[key]

    def __iter__(self):
        return iter(self._files)

    def __len__(self) -> int:
        return len(self._files)
