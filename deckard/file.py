from __future__ import annotations

import time
from typing import Any, TypedDict
from uuid import uuid4

from hydra.core.hydra_config import HydraConfig

# -----------------------------------------------------------------------------
# TypedDict definitions unchanged
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# error
# -----------------------------------------------------------------------------


class FileConfigError(TypeError):
    pass


# -----------------------------------------------------------------------------
# resolver
# -----------------------------------------------------------------------------


class PlaceholderResolverMixin:
    replace: dict[str, str]

    @property
    def num(self) -> str:
        """Returns the serial job number in a multirun sweep. Uses uuid as fallback if Hydra is not enabled."""
        try:
            return str(HydraConfig.get().job.num)
        except ValueError:
            return uuid4().hex

    @property
    def id(self) -> str:
        """Returns the specific launcher or cluster job ID. Uses uuid as fallback if Hydra is not enabled"""
        try:
            return str(HydraConfig.get().job.num)
        except ValueError:
            return uuid4().hex

    def _resolve(self, value: str | None) -> str | None:
        if not value:
            return None

        value = value.replace("{num}", self.num)
        value = value.replace("{#}", self.num)
        value = value.replace("{timestamp}", time.strftime("%Y%m%d-%H%M%S"))
        value = value.replace("{hash}", self.id)
        value = value.replace("{*}", self.id)

        for k, v in getattr(self, "replace", {}).items():
            value = value.replace(k, str(v))

        return value


# -----------------------------------------------------------------------------
# FIXED FILE CONFIG
# -----------------------------------------------------------------------------
class FileConfig(PlaceholderResolverMixin):
    """
    Dynamic file-path configuration container.

    `FileConfig` manages resolved artifact paths for datasets, models,
    predictions, logs, attacks, and scores.

    File fields are validated against the configured file schema and stored
    internally in `_files`.

    Placeholder expansion is applied automatically to all string values.

    Supported placeholders:

    - `{num}` → Hydra job number (`HYDRA_JOB_NUM`, default `0`)
    - `{timestamp}` → current timestamp (`YYYYMMDD-HHMMSS`)
    - `{hash}` → hash of the `FileConfig` instance
    - `#` → alias for `{num}`
    - `*` → alias for `{num}`

    User-defined replacements may also be provided through `replace`.

    Example
    -------

    ```python
    config = FileConfig(
        replace={"{exp}": "demo"},
        model_file="models/{exp}/{hash}.pt",
        attack_file="attacks/#/*.json",
    )

    print(config.model_file)
    print(config.attack_file)
    ```

    Parameters
    ----------
    replace
        Optional placeholder replacement mapping.

    **files
        File-path keyword arguments matching the configured schema, such as
        `data_file`, `model_file`, `log_file`, or `attack_file`.

    Raises
    ------
    FileConfigError
        Raised when an unknown file key is provided.

    Notes
    -----
    This class separates:

    - **schema layer**: `TypedDict` definitions for IDE support
    - **runtime layer**: validated dynamic file storage
    - **resolution layer**: placeholder substitution

    `TypedDict` definitions are used only for static typing and validation.
    Runtime file values are stored in `_files`.
    """

    def __init__(self, *, replace: dict[str, str] | None = None, **files: Any):
        self.replace = replace or {}
        self._files: dict[str, Any] = {}

        for k, v in files.items():
            self._validate_key(k)
            self._set(k, v)

    # -------------------------------------------------------------------------
    # validation
    # -------------------------------------------------------------------------

    def _validate_key(self, key: str) -> None:
        if key not in _ALLOWED_KEYS:
            raise FileConfigError(f"Invalid file key: {key}")

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
        for k, v in kwargs.items():
            self._validate_key(k)
            self._set(k, v)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._files)

    def __getitem__(self, key: str) -> Any:
        return self._files[key]

    def __iter__(self):
        return iter(self._files)

    def __len__(self) -> int:
        return len(self._files)
