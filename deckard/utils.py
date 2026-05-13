"""Shared utility layer for deckard's public Python API.

This module contains the base configuration protocol used across the project,
stable hashing helpers for config identity, file IO helpers, and utility
functions for dynamically resolving and instantiating classes.
"""

import logging
import argparse
import inspect
import pandas as pd
import pickle
import json
import importlib
import importlib.util
import sys
import traceback
import hashlib

from pathlib import Path
from typing import Iterable, Optional, Union, Any
from dataclasses import MISSING, dataclass, field
from hydra.utils import instantiate, get_class
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigBase",
    "normalize_config_token",
    "is_null_config_value",
    "is_default_config_value",
    "normalize_for_hash",
    "hash_conf_values",
    "data_supported_filetypes",
    "save_data",
    "load_data",
    "import_class_from_file",
    "resolve_class",
    "load_class",
    "safe_store",
    "coerce_config",
    "prepare_instantiation_dict",
    "instantiate_config",
    "create_parser_from_function",
    "round_scores",
    "coerce_to_list",
    "normalize_optional_list_value",
    "normalize_optional_mapping_or_steps",
    "normalize_plugin_specs",
    "instantiate_plugin_spec",
    "split_comma_separated_tokens",
    "normalize_hydra_list_overrides",
    "merge_list_of_dicts",
    "merge_scores_with_collision_suffix",
    "resolve_torch_device",
    "probabilities_from_model_outputs",
]


NULL_CONFIG_TOKENS = frozenset({"none", "null", "n/a"})
DEFAULT_CONFIG_TOKENS = frozenset({"auto", "default", "best"})


def normalize_config_token(value: Any) -> str | None:
    """Normalize a config token to lower-case text for keyword matching."""
    if value is None:
        return None
    return str(value).strip().lower()


def is_null_config_value(value: Any, *, allow_empty: bool = True) -> bool:
    """Return True when *value* represents an explicit null-like config token."""
    token = normalize_config_token(value)
    if token is None:
        return True
    if allow_empty and token == "":
        return True
    return token in NULL_CONFIG_TOKENS


def is_default_config_value(value: Any, *, include_best: bool = True) -> bool:
    """Return True when *value* requests default/auto config behavior."""
    token = normalize_config_token(value)
    if token is None:
        return False
    if include_best:
        return token in DEFAULT_CONFIG_TOKENS
    return token in {"auto", "default"}


def _torch_compiler_backends(torch_module) -> list[str]:
    compiler = getattr(torch_module, "compiler", None)
    if compiler is None or not hasattr(compiler, "list_backends"):
        return []
    try:
        return [str(name).strip().lower() for name in compiler.list_backends()]
    except Exception:
        return []


def _auto_torch_device_from_backends(torch_module):
    backends = set(_torch_compiler_backends(torch_module))
    cuda_available = bool(torch_module.cuda.is_available())
    mps_available = bool(
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available(),
    )

    cuda_preferred = {
        "inductor",
        "cudagraphs",
        "onnxrt",
        "openxla",
        "tensorrt",
    }
    mps_preferred = {"inductor", "aot_eager", "eager"}

    if cuda_available and len(backends.intersection(cuda_preferred)) > 0:
        logger.info(
            "Auto-selected torch device cuda:0 via compiler backends: %s",
            sorted(backends),
        )
        return torch_module.device("cuda:0")
    if mps_available and len(backends.intersection(mps_preferred)) > 0:
        logger.info(
            "Auto-selected torch device mps via compiler backends: %s",
            sorted(backends),
        )
        return torch_module.device("mps")
    if cuda_available:
        logger.info("Auto-selected torch device cuda:0")
        return torch_module.device("cuda:0")
    if mps_available:
        logger.info("Auto-selected torch device mps")
        return torch_module.device("mps")
    logger.info("Auto-selected torch device cpu")
    return torch_module.device("cpu")


def resolve_torch_device(requested_device: Any = None) -> Any:
    try:
        import torch
    except ImportError:
        if requested_device is None:
            return "cpu"
        return str(requested_device)

    if isinstance(requested_device, torch.device):
        return requested_device

    if isinstance(requested_device, int):
        if (
            requested_device >= 0
            and torch.cuda.is_available()
            and requested_device < torch.cuda.device_count()
        ):
            return torch.device(f"cuda:{requested_device}")
        fallback = _auto_torch_device_from_backends(torch)
        logger.warning(
            "Invalid CUDA index %s; falling back to best available device %s",
            requested_device,
            fallback,
        )
        return fallback

    requested_text = ""
    if requested_device is not None:
        requested_text = str(requested_device).strip().lower()

    if is_null_config_value(requested_device) or is_default_config_value(
        requested_device,
        include_best=True,
    ):
        return _auto_torch_device_from_backends(torch)
    if requested_text in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        fallback = _auto_torch_device_from_backends(torch)
        logger.warning(
            "GPU requested but CUDA unavailable; falling back to best available device %s",
            fallback,
        )
        return fallback
    if requested_text.startswith("cuda"):
        if not torch.cuda.is_available():
            fallback = _auto_torch_device_from_backends(torch)
            logger.warning(
                "CUDA requested but unavailable; falling back to best available device %s",
                fallback,
            )
            return fallback
        try:
            return torch.device(requested_text)
        except (TypeError, ValueError, RuntimeError):
            fallback = _auto_torch_device_from_backends(torch)
            logger.warning(
                "Invalid CUDA device '%s'; falling back to best available device %s",
                requested_text,
                fallback,
            )
            return fallback
    if requested_text.startswith("mps"):
        mps_available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        )
        if not mps_available:
            fallback = _auto_torch_device_from_backends(torch)
            logger.warning(
                "MPS requested but unavailable; falling back to best available device %s",
                fallback,
            )
            return fallback
        return torch.device("mps")

    try:
        return torch.device(requested_text)
    except (TypeError, ValueError, RuntimeError):
        logger.warning(
            "Invalid torch device '%s'; falling back to auto device",
            requested_text,
        )
        return _auto_torch_device_from_backends(torch)


def safe_store(group: str, name: str, node: Any) -> None:
    """Register a Hydra config node while tolerating duplicate registrations."""
    cs = ConfigStore.instance()
    try:
        cs.store(group=group, name=name, node=node)
    except Exception:
        # Re-imports in tests/dev can register the same node repeatedly.
        pass


def coerce_to_list(items: Union[list, Any]) -> list:
    """Normalize a ``list`` or OmegaConf ``ListConfig`` to a plain Python list.

    Parameters
    ----------
    items:
        A ``list`` or OmegaConf ``ListConfig``.

    Returns
    -------
    list
        A plain Python list whose elements are the same objects as *items*.

    Raises
    ------
    TypeError
        If *items* is neither a ``list`` nor a ``ListConfig``.
    """
    from omegaconf import ListConfig

    if isinstance(items, (list, ListConfig)):
        return list(items)
    raise TypeError(f"Expected list or ListConfig, got {type(items)}")


def normalize_optional_list_value(
    value: Any,
    *,
    field_name: str = "value",
) -> Optional[list]:
    """Normalize optional list-like config values to ``list`` or ``None``.

    Accepts null-like tokens, plain strings, ``list``, and OmegaConf
    ``ListConfig`` values.
    """
    if is_null_config_value(value):
        return None
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return list(value)
    raise TypeError(
        f"{field_name} expected optional list-like value, got {type(value)}",
    )


def normalize_optional_mapping_or_steps(
    value: Any,
    *,
    field_name: str,
) -> Union[None, bool, dict[str, Any]]:
    """Normalize optional mapping-or-step-list config values.

    - ``None`` / null-like tokens -> ``None``
    - ``True`` / ``False`` -> same bool
    - ``dict`` / ``DictConfig`` -> plain ``dict``
    - ``list`` / ``ListConfig`` -> merged dict via ``merge_list_of_dicts``
    """
    if isinstance(value, (list, ListConfig)):
        return merge_list_of_dicts(coerce_to_list(value))
    if isinstance(value, bool):
        return value
    if is_null_config_value(value):
        return None
    if isinstance(value, DictConfig):
        return dict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(
        f"{field_name} must be None, bool, dict/DictConfig, or list/ListConfig. Got {type(value)}",
    )


def normalize_plugin_specs(plugins: Any) -> list:
    """Normalize plugin config values into a plain list.

    Accepts ``None`` / null-like tokens, single plugin specs, ``list`` and
    OmegaConf ``ListConfig`` values.
    """
    if is_null_config_value(plugins):
        return []
    if isinstance(plugins, ListConfig):
        return list(plugins)
    if isinstance(plugins, list):
        return list(plugins)
    return [plugins]


def instantiate_plugin_spec(
    plugin_spec: Any,
    *,
    loader,
) -> Any:
    """Instantiate one plugin spec using a caller-provided loader.

    ``loader`` must accept ``loader(path, **kwargs)`` and return an instance.
    """
    if isinstance(plugin_spec, dict):
        spec = dict(plugin_spec)
        class_path = spec.pop("name", spec.pop("_target_", None))
        if class_path is None:
            raise ValueError("Plugin dict must include 'name' or '_target_'")
        return loader(class_path, **spec)

    if isinstance(plugin_spec, str):
        return loader(plugin_spec)

    if isinstance(plugin_spec, type):
        return plugin_spec()

    return plugin_spec


def split_comma_separated_tokens(value: Any) -> list[str]:
    """Split a comma-separated string into trimmed, non-empty tokens."""
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip() != ""]


def normalize_hydra_list_overrides(
    overrides: list[str],
    *,
    keys: tuple[str, ...] = ("score",),
) -> list[str]:
    """Rewrite comma-separated Hydra override values into list syntax.

    Example: ``score=a,b`` becomes ``score=[a,b]``.
    Existing bracketed list syntax is preserved unchanged.
    """
    normalized: list[str] = []
    normalized_keys = {str(key).strip() for key in keys}
    for token in overrides:
        if not isinstance(token, str) or "=" not in token:
            normalized.append(token)
            continue

        key, value = token.split("=", 1)
        if key not in normalized_keys:
            normalized.append(token)
            continue

        stripped = value.strip()
        if "," not in stripped:
            normalized.append(token)
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            normalized.append(token)
            continue

        items = split_comma_separated_tokens(stripped)
        normalized.append(f"{key}=[{','.join(items)}]")

    return normalized


def merge_list_of_dicts(items: Iterable) -> dict:
    """Merge a sequence of dict-like items into a single ``dict``.

    Each element of *items* may be a plain ``dict`` or an OmegaConf
    ``DictConfig``; the latter is converted via ``OmegaConf.to_container``
    before merging.  When two elements share a key, the later element wins.

    Parameters
    ----------
    items:
        An iterable of ``dict`` / ``DictConfig`` objects.

    Returns
    -------
    dict
        A single merged plain-Python ``dict``.

    Raises
    ------
    TypeError
        If any element cannot be resolved to a ``dict``.
    """
    merged: dict = {}
    for item in items:
        if OmegaConf.is_config(item):
            item = OmegaConf.to_container(item, resolve=True)
        if not isinstance(item, dict):
            raise TypeError(
                f"merge_list_of_dicts: each element must be a dict, got {type(item)}",
            )
        merged.update(item)
    return merged


def merge_scores_with_collision_suffix(
    base_scores: dict,
    incoming_scores: dict,
    *,
    alias: Optional[str] = None,
) -> dict:
    """Merge score dictionaries while preserving existing keys.

    Behavior
    --------
    - Non-colliding keys are copied as-is.
    - Colliding keys are only suffixed when ``alias`` is provided.
    - If ``alias`` is ``None``, incoming collisions receive an underscore and an increment (e.g._1 for the 2nd key of the same name).
    """
    if not isinstance(base_scores, dict):
        raise TypeError(f"base_scores must be a dict, got {type(base_scores)}")
    if not isinstance(incoming_scores, dict):
        raise TypeError(
            f"incoming_scores must be a dict, got {type(incoming_scores)}",
        )

    merged = dict(base_scores)
    for key, value in incoming_scores.items():
        if key not in merged:
            merged[key] = value
            continue

        if alias is None:
            merged[key] = value
            continue

        candidate = f"{key}_{alias}"
        if candidate not in merged:
            merged[candidate] = value
            continue

        disambiguation_index = 2
        while f"{candidate}_{disambiguation_index}" in merged:
            disambiguation_index += 1
        merged[f"{candidate}_{disambiguation_index}"] = value

    return merged


def coerce_config(config_obj: Any) -> Any:
    """Coerce config-like objects into plain Python structures when possible.

    Supported coercions:
    - ``DictConfig`` -> ``dict``/``list`` via ``OmegaConf.to_container``
    - ``ConfigBase`` -> ``dict`` via ``to_dict``
    - existing YAML file path string -> loaded config container
    """
    if config_obj is None:
        return None

    if isinstance(config_obj, DictConfig):
        return OmegaConf.to_container(config_obj, resolve=True)

    if isinstance(config_obj, ConfigBase):
        return config_obj.to_dict()

    if isinstance(config_obj, str):
        path = Path(config_obj)
        if path.exists() and path.suffix in {".yaml", ".yml"}:
            return OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    return config_obj


def prepare_instantiation_dict(
    config_obj: Any,
    *,
    default_target: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize config-like input into an instantiation dictionary.

    When ``default_target`` is provided and the resulting mapping lacks a
    ``_target_`` key, the target is injected so Hydra can instantiate the
    canonical config class while still allowing user-specified overrides.
    """
    config_obj = coerce_config(config_obj)
    if not isinstance(config_obj, dict):
        raise TypeError(
            "Config must resolve to a dict-like object for instantiation. "
            f"Got {type(config_obj)}",
        )
    spec = dict(config_obj)
    if default_target is not None and "_target_" not in spec:
        spec["_target_"] = default_target
    return spec


def instantiate_config(
    config_obj: Any,
    expected_type: type,
    *,
    default_target: Optional[str] = None,
) -> Any:
    """Instantiate one config object, injecting a default Hydra target when needed."""
    if config_obj is None:
        return None
    if isinstance(config_obj, expected_type):
        return config_obj

    if default_target is None:
        default_target = f"{expected_type.__module__}.{expected_type.__name__}"

    spec = prepare_instantiation_dict(
        config_obj,
        default_target=default_target,
    )
    instance = instantiate(spec)
    if not isinstance(instance, expected_type):
        raise TypeError(
            f"Expected instantiated config to be {expected_type.__name__}, "
            f"got {type(instance)}",
        )
    return instance


def round_scores(
    scores: dict,
    n_samples: int,
    logger_obj: Optional[logging.Logger] = None,
) -> dict:
    """Round numeric score values using a sample-size-aware precision rule.

    The number of decimal places is derived from ``log10(n_samples) + 1`` and
    clamped to at least one decimal place.
    """
    if n_samples is None or n_samples <= 0:
        sig_figs = 1
    else:
        sig_figs = np.log10(n_samples) + 1
        if sig_figs < 1:
            sig_figs = 1
    decimals = int(sig_figs)

    if logger_obj is not None:
        logger_obj.info(f"Rounding scores to {decimals} significant figures")
        logger_obj.info("Scores:")

    rounded_scores = dict(scores)
    for score, value in list(rounded_scores.items()):
        if isinstance(value, (int, float, np.integer, np.floating)):
            rounded = round(float(value), decimals)
            rounded_scores[score] = rounded
            if logger_obj is not None:
                logger_obj.info(f"{score}: {rounded}")
        elif logger_obj is not None:
            logger_obj.info(f"{score}: {value}")
    return rounded_scores


def probabilities_from_model_outputs(outputs: Any) -> np.ndarray:
    """Convert classifier outputs into a 2D probability array.

    Accepts logits/probabilities from torch or numpy outputs and returns
    ``(n_samples, n_classes)`` probabilities for downstream scorers.
    """
    if hasattr(outputs, "detach") and hasattr(outputs, "cpu"):
        arr = outputs.detach().cpu().numpy()
    else:
        arr = np.asarray(outputs)

    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr.astype(np.float64, copy=False)
        max_per_row = np.max(arr, axis=1, keepdims=True)
        exp_logits = np.exp(arr - max_per_row)
        denom = np.sum(exp_logits, axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return exp_logits / denom

    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)

    if arr.ndim == 1:
        arr = arr.astype(np.float64, copy=False)
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            pos = 1.0 / (1.0 + np.exp(-arr))
        else:
            pos = np.clip(arr, 0.0, 1.0)
        neg = 1.0 - pos
        return np.column_stack([neg, pos])

    raise ValueError(
        f"Unable to derive probability predictions from output shape {arr.shape}",
    )


def _canonicalize_for_hash(value):
    """Convert arbitrary values into a stable, JSON-serializable structure."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"

    if isinstance(value, dict):
        return {
            str(k): _canonicalize_for_hash(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_hash(v) for v in value]

    if isinstance(value, (set, frozenset)):
        items = [_canonicalize_for_hash(v) for v in value]
        return sorted(
            items,
            key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")),
        )

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": bytes(value).hex()}

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _canonicalize_for_hash(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        public_attrs = {
            k: v
            for k, v in value.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
        if not public_attrs:
            return str(value)
        return _canonicalize_for_hash(public_attrs)

    return str(value)


def normalize_for_hash(value: Any, root: Optional[Any] = None) -> Any:
    """Normalize values for stable hashing.

    Mirrors resolver behavior:
    - Optional key-path lookup when `value` is a string and `root` is provided.
    - OmegaConf nodes resolved to plain Python containers.
    """
    target = value

    if isinstance(value, str) and root is not None:
        selected = OmegaConf.select(root, value, default=None)
        if selected is not None:
            target = selected

    return _canonicalize_for_hash(target)


def hash_conf_values(*values, _root_=None) -> str:
    """Return stable MD5 hash for one or more config-like values.

    Supports the same patterns as the `${hash:...}` resolver:
    - no values: hash `_root_`
    - one value: hash normalized value
    - many values: hash normalized list of values in order
    """
    if not values:
        target = _root_
    elif len(values) == 1:
        target = normalize_for_hash(values[0], root=_root_)
    else:
        target = [normalize_for_hash(v, root=_root_) for v in values]

    s = json.dumps(target, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


data_supported_filetypes = [
    ".csv",
    ".parquet",
    ".pkl",
    ".html",
    ".json",
    ".xlsx",
    ".openml",
]


@dataclass
class ConfigBase:
    """Base class for deckard configuration objects.

    ``ConfigBase`` provides a common lifecycle for config dataclasses: argument
    hydration, post-init hooks, stable hashing based on configuration state, and
    serialization helpers used throughout deckard.
    """

    # _target_: str = "deckard.utils.ConfigBase"
    score_dict: dict = field(default_factory=dict)
    HASH_EXCLUDE_FIELDS = {
        "args",
        "score_dict",
        "predictions",
        "probabilities",
        "labels",
        "attack_predictions",
        "attack_probabilities",
        "adv_predictions",
        "adv_probabilities",
        "X",
        "y",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
    }
    HASH_EXCLUDE_SUFFIXES = (
        "_time",
        "_predictions",
        "_probabilities",
    )

    def __init__(self, *args, **kwds):
        # Initialize dataclass super
        super().__init__()

        # Initialize args attribute
        self.args = args if args else ()

        dataclass_fields = self.__dataclass_fields__
        init_fields = [
            field_name
            for field_name, dataclass_field in dataclass_fields.items()
            if dataclass_field.init
        ]

        # Seed dataclass defaults/default_factories before applying user values.
        for field_name, dataclass_field in dataclass_fields.items():
            if dataclass_field.default is not MISSING:
                setattr(self, field_name, dataclass_field.default)
            elif dataclass_field.default_factory is not MISSING:
                setattr(self, field_name, dataclass_field.default_factory())

        if len(args) > len(init_fields):
            raise TypeError(
                f"Expected at most {len(init_fields)} positional arguments, got {len(args)}",
            )

        for i, arg in enumerate(args):
            setattr(self, init_fields[i], arg)
        for k, v in kwds.items():
            setattr(self, k, v)

        self._before_post_init()
        self.__post_init__()
        self._after_post_init()

    def __post_init__(self):
        pass

    def _before_post_init(self) -> None:
        """Hook for subclasses that need pre-normalization before __post_init__."""

    def _after_post_init(self) -> None:
        """Finalize common lifecycle state after __post_init__."""
        # Freeze hash at configuration time so runtime attributes added during
        # execution cannot alter experiment identity.
        self._hash_payload = self.to_dict(for_hash=True)
        self._hash_value = hash_conf_values(self._hash_payload)

    def coerce_component(
        self,
        component: Any,
        expected_type: type,
        *,
        default_target: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
        allow_passthrough: Optional[Any] = None,
    ) -> Any:
        """Instantiate/normalize a config component to ``expected_type``.

        Parameters
        ----------
        component : Any
            Config-like object, runtime object, or expected config instance.
        expected_type : type
            The required config class.
        default_target : str, optional
            Hydra target path used when input lacks ``_target_``.
        overrides : dict, optional
            Key/value pairs injected into the normalized instantiation spec.
        allow_passthrough : callable, optional
            If provided and returns ``True`` for *component*, bypasses coercion.
        """
        if component is None:
            return None

        if isinstance(component, expected_type):
            return component

        if allow_passthrough is not None and allow_passthrough(component):
            return component

        if default_target is None:
            default_target = f"{expected_type.__module__}.{expected_type.__name__}"

        spec = prepare_instantiation_dict(component, default_target=default_target)
        if overrides:
            spec.update(overrides)

        instance = instantiate(spec)
        if not isinstance(instance, expected_type):
            raise TypeError(
                f"Expected instantiated config to be {expected_type.__name__}, "
                f"got {type(instance)}",
            )
        return instance

    def __call__(self):
        raise NotImplementedError("This is an abstract base class.")

    def __hash__(self):
        """Return the initialization-time configuration hash as int."""
        if "_hash_value" not in self.__dict__:
            self._hash_payload = self.to_dict(for_hash=True)
            self._hash_value = hash_conf_values(self._hash_payload)
        return int(self._hash_value, 16)

    def __eq__(self, other: object) -> bool:
        """Two ConfigBase instances are equal when their configuration hashes match."""
        if not isinstance(other, ConfigBase):
            return NotImplemented
        return hash(self) == hash(other)

    def _is_hash_field(self, name: str) -> bool:
        if name == "_target_":
            return True
        if name.startswith("_"):
            return False
        if name in self.HASH_EXCLUDE_FIELDS:
            return False
        if any(name.endswith(suffix) for suffix in self.HASH_EXCLUDE_SUFFIXES):
            return False
        return True

    def save_scores(
        self,
        scores: Union[dict, pd.Series],
        filepath: Optional[str] = None,
    ) -> None:
        """
        Saves the scores dictionary to a CSV file if a filepath is provided.

        Parameters
        ----------
        scores : dict
            Dictionary containing score metrics to be saved.
        filepath : Union[str, None], optional
            Path to save the scores as a CSV file. If None, scores are not saved.

        Raises
        ----------
        ValueError
            If the file extension is not supported. Supported types are .csv, .json, and .xlsx.
        """
        assert filepath is not None, "Filepath must be provided to save scores."
        score_path = Path(filepath)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        # Assume this is a dictionary of of strings: floats
        supported_filtypes = [".csv", ".json", ".xlsx"]
        if not isinstance(scores, dict):
            scores = dict(scores)
        if score_path.suffix in supported_filtypes:
            match score_path.suffix:
                case ".csv":
                    pd.DataFrame([scores]).to_csv(score_path, index=False)
                case ".json":
                    with open(score_path, "w") as f:
                        json.dump(scores, f, indent=4)
                case ".xlsx":
                    pd.DataFrame([scores]).to_excel(score_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filtypes}",
            )
        assert Path(
            score_path,
        ).exists(), f"Failed to save scores to {score_path}"
        logger.info(f"Scores saved to {score_path}")

    def save_data(
        self,
        data: pd.DataFrame,
        filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        supported_filetypes = [
            ".csv",
            ".parquet",
            ".pkl",
            ".html",
            ".json",
            ".xlsx",
        ]
        assert filepath is not None, "Filepath must be provided to save data."
        data_path = Path(filepath)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        filetype = data_path.suffix
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        match filetype:
            case ".pkl":
                data.to_pickle(data_path, **kwargs)
            case ".csv":
                data.to_csv(data_path, index=False, **kwargs)
            case ".parquet":
                data.to_parquet(data_path, index=False, **kwargs)
            case ".pkl":
                data.to_pickle(data_path, **kwargs)
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
        logger.info(f"Data saved to {data_path}")

    def read_or_initialize_scores(self, score_file: Optional[str]) -> dict:
        """Return merged scores from disk and memory, or initialize output location.

        This is the canonical entrypoint for score-file reads in ConfigBase.
        """
        if score_file is not None and Path(score_file).exists():
            # Load existing scores
            logger.info(f"Loading existing scores from {score_file}")
            disk_scores = self.load_scores(score_file)
            scores = {**self.score_dict, **disk_scores}
        elif score_file is not None:
            # Ensure directory exists
            logger.debug(f"Creating directory for scores at {score_file}")
            Path(score_file).parent.mkdir(parents=True, exist_ok=True)
            scores = self.score_dict
        else:
            logger.debug("No score_file provided, scores will not be saved")
            if hasattr(self, "score_dict"):
                scores = self.score_dict
            else:
                scores = {}
        return scores

    def merge_and_persist_scores(
        self,
        new_scores: dict,
        score_file: Optional[str],
    ) -> dict:
        """Merge score payload with on-disk scores and persist only when needed."""
        if score_file is None:
            return new_scores

        score_path = Path(score_file)
        score_path.parent.mkdir(parents=True, exist_ok=True)

        existing_scores: dict = {}
        if score_path.exists():
            existing_scores = self.load_scores(score_file)

        merged_scores = {**existing_scores, **new_scores}
        if (not score_path.exists()) or merged_scores != existing_scores:
            self.save_scores(merged_scores, score_file)
        return merged_scores

    def get_call_params(self) -> dict:
        """
        Retrieves the parameters required to call the __call__ method of the instance.

        Returns
        -------
        dict
            A dictionary containing parameter names and their corresponding values.
        """
        sig = inspect.signature(self.__call__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
            else:
                raise AttributeError(
                    f"Instance of {self.__class__.__name__} does not have attribute {name} required for __call__",
                )
        return params

    def load_scores(self, filepath: str) -> dict:
        """
        Loads scores from a CSV, JSON, or Excel file into a dictionary.

        Parameters
        ----------
        filepath : str
            Path to the scores file.

        Returns
        -------
        dict
            Dictionary containing the loaded scores.

        Raises
        ------
        ValueError
            If the file extension is not supported. Supported types are .csv, .json, and .xlsx.
        """
        score_path = Path(filepath)
        assert score_path.exists(), f"File {filepath} does not exist."
        supported_filetypes = [".csv", ".json", ".xlsx"]
        scores: dict
        if score_path.suffix in supported_filetypes:
            match score_path.suffix:
                case ".csv":
                    df = pd.read_csv(score_path)
                    if len(df) == 0:
                        scores = {}
                    else:
                        scores = df.iloc[0].to_dict()
                case ".json":
                    with open(score_path, "r") as f:
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
                    if len(df) == 0:
                        scores = {}
                    else:
                        scores = df.iloc[0].to_dict()
                case _:
                    raise ValueError(
                        f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
                    )
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
            )
        logger.info(f"Scores loaded from {score_path}")
        return {str(k): v for k, v in scores.items()}

    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        return load_data(filepath, **kwargs)

    def save_object(self, obj: Any, filepath: str) -> None:
        """
        Saves a Serializable object to a file using pickle.

        Parameters
        ----------
        obj : Any
            The object to save.
        filepath : str
            The path to the file where the object will be saved.
        Raises
        ------
        ValueError
            If the file extension is not supported. Supported types are .pkl and .pickle.
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
        logger.info(f"Object saved to {filepath}")

    def load_object(
        self,
        filepath: str,
        ignore_corrupt: bool = False,
        delete_corrupt: bool = False,
    ) -> Any:
        """
        Loads a Serializable object from a file using pickle.

        Parameters
        ----------
        filepath : str
            The path to the file from which the object will be loaded.

        Returns
        -------
        Any
            The loaded object.

        Parameters
        ----------
        ignore_corrupt : bool, optional
            If True, return None when a known pickle corruption/loading error occurs.
        delete_corrupt : bool, optional
            If True and a known pickle corruption/loading error occurs, delete the file.
        """
        try:
            with open(filepath, "rb") as f:
                obj = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, AttributeError, OSError) as exc:
            if delete_corrupt:
                Path(filepath).unlink(missing_ok=True)
            if ignore_corrupt:
                logger.warning(
                    "Failed to load cached object %s (%s).",
                    filepath,
                    exc,
                )
                return None
            raise
        logger.info(f"Object loaded from {filepath}")
        return obj

    def save(self, filepath: str) -> None:
        """
        Saves the current instance to a file using pickle.

        Parameters
        ----------
        filepath : str
            The path to the file where the instance will be saved.
        """
        self.save_object(self, filepath)
        logger.info(
            f"Instance of {self.__class__.__name__} saved to {filepath}",
        )

    def load(self, filepath: str) -> "ConfigBase":
        """
        Loads an instance of the class from a file using pickle.

        Parameters
        ----------
        filepath : str
            The path to the file from which the instance will be loaded.

        Returns
        -------
        ConfigBase
            The loaded instance.
        """
        assert Path(filepath).exists(), f"File {filepath} does not exist."
        obj = self.load_object(filepath)
        if not isinstance(obj, self.__class__):
            raise TypeError(
                f"Loaded object is not of type {self.__class__.__name__}",
            )
        logger.info(
            f"Instance of {self.__class__.__name__} loaded from {filepath}",
        )
        # Update the current instance's __dict__ with the loaded object's __dict__
        self.__dict__.update(obj.__dict__)
        return self

    @staticmethod
    def from_yaml(filepath: str) -> "ConfigBase":
        """
        Creates an instance of the class from a YAML configuration file.

        Parameters
        ----------
        filepath : str
            The path to the YAML configuration file.

        Returns
        -------
        ConfigBase
            An instance of the class initialized with the configuration from the YAML file.
        """
        config = OmegaConf.to_container(OmegaConf.load(filepath), resolve=True)
        if not isinstance(config, dict):
            raise TypeError(
                f"Loaded config is not a dictionary from {filepath}",
            )
        instance = instantiate(config)
        logger.info(
            f"Instance of {instance.__class__.__name__} created from {filepath}",
        )
        return instance

    @staticmethod
    def from_dict(data: dict) -> "ConfigBase":
        """
        Creates an instance of the class from a dictionary.

        Parameters
        ----------
        data : dict
            The dictionary containing the configuration.

        Returns
        -------
        ConfigBase
            An instance of the class initialized with the configuration from the dictionary.
        """
        instance = instantiate(data)
        return instance

    def to_yaml(self) -> str:
        """
        Converts the current instance to a YAML string.

        Returns
        -------
        str
            A YAML representation of the instance.
        """
        config = self.to_dict()
        config = OmegaConf.create(config)
        return str(OmegaConf.to_yaml(config))

    def to_dict(self, for_hash: bool = False) -> dict:
        """
        Converts the current instance to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the instance.
        """
        # Build a dict from inherited dataclass fields + runtime attributes
        dict_ = {}

        # Include dataclass fields from full MRO (base -> child)
        for base in reversed(self.__class__.mro()):
            fields = getattr(base, "__dataclass_fields__", {})
            for name in fields:
                if name.startswith("_") and not (for_hash and name == "_target_"):
                    continue
                if for_hash and not self._is_hash_field(name):
                    continue
                if hasattr(self, name):
                    value = getattr(self, name)
                    if isinstance(value, ConfigBase):
                        dict_[name] = value.to_dict(for_hash=for_hash)
                    elif OmegaConf.is_config(value):
                        dict_[name] = OmegaConf.to_container(
                            value,
                            resolve=True,
                        )
                    else:
                        dict_[name] = value

        # Include any additional runtime attrs not declared as dataclass fields
        for name, value in self.__dict__.items():
            if (
                name.startswith("_") and not (for_hash and name == "_target_")
            ) or name in dict_:
                continue
            if for_hash and not self._is_hash_field(name):
                continue
            if isinstance(value, ConfigBase):
                dict_[name] = value.to_dict(for_hash=for_hash)
            elif OmegaConf.is_config(value):
                dict_[name] = OmegaConf.to_container(value, resolve=True)
            else:
                dict_[name] = value

        return dict_

    def execute_without_mercy(self) -> dict:
        # Get log_file from logger
        log_file = next(
            (
                handler.baseFilename
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ),
            "deckard.log",
        )
        try:
            scores = self()
        except Exception as e:
            with open(log_file, "+a") as log_f:
                tb = traceback.format_exc()
                log_f.write(f"\nException: {e}\n")
                log_f.write(tb)
                log_f.write("\n")
            logger.error(e)
            if hasattr(self, "score_dict"):
                scores = self.score_dict
            else:
                scores = {}
        return scores


def save_data(
    data: pd.DataFrame,
    filepath: Union[str, None] = None,
    **kwargs,
) -> None:
    """Persist tabular data to one of deckard's supported file formats."""
    supported_filetypes = [
        ".csv",
        ".parquet",
        ".pkl",
        ".html",
        ".json",
        ".xlsx",
    ]
    assert filepath is not None, "Filepath must be provided to save data."
    data_path = Path(filepath)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    filetype = data_path.suffix
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
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
    logger.info(f"Data saved to {data_path}")


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Loads data from a CSV, JSON, Excel, Parquet, Pickle, NPZ, or HTML file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    **kwargs
        Additional keyword arguments to pass to the pandas read function.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file extension is not supported. Supported types are .csv, .json, .
    """

    if filepath is None:
        raise FileNotFoundError("Filepath is None.")
    supported_filetypes = [
        ".csv",
        ".json",
        ".xlsx",
        ".parquet",
        ".pkl",
        ".npz",
        ".html",
    ]

    match Path(filepath).suffix:
        case ".pkl":
            data = pd.read_pickle(filepath, **kwargs)
        case ".csv":
            data = pd.read_csv(filepath, **kwargs)
        case ".json":
            json_kwargs = {"orient": "records", **kwargs}
            if "lines" not in json_kwargs:
                try:
                    data = pd.read_json(filepath, lines=True, **json_kwargs)
                except ValueError:
                    data = pd.read_json(filepath, **json_kwargs)
            else:
                data = pd.read_json(filepath, **json_kwargs)
        case ".xlsx":
            data = pd.read_excel(filepath, **kwargs)
        case ".parquet":
            data = pd.read_parquet(filepath, **kwargs)
        case ".html":
            data = pd.read_html(filepath, **kwargs)[0]
        case _:
            raise ValueError(
                f"Unsupported file type {Path(filepath).suffix}. Supported types: {supported_filetypes}",
            )
    logger.info(f"Data loaded from {Path(filepath)}")
    return data


def import_class_from_file(
    file_path: Union[str, Path],
    class_name: str,
    *args: Any,
    instantiate_class: bool = True,
    **kwargs: Any,
) -> Any:
    """Import a class from a Python file path and optionally instantiate it."""
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    spec.loader.exec_module(module)

    cls = getattr(module, class_name)
    if not instantiate_class:
        return cls
    return cls(*args, **kwargs)


def resolve_class(cls: str) -> Any:
    """Resolve a class path into a class object without instantiating it.

    Supports dotted module paths (Hydra-style) and ``file.py:ClassName`` paths.
    """
    if not isinstance(cls, str):
        raise TypeError(f"Class path must be a string. Got {type(cls)}")

    if ":" in cls:
        file_path, class_name = cls.split(":", 1)
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        return import_class_from_file(
            file_path,
            class_name,
            instantiate_class=False,
        )

    module_name, attr_name = cls.rsplit(".", 1)

    # Prefer direct module attribute resolution first. This supports classes
    # and functions without emitting Hydra's "non-class" diagnostics.
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except Exception:
        # Fall back to Hydra's class resolver for non-standard targets.
        return get_class(cls)


def load_class(cls: Union[str, type], *args: Any, **kwargs: Any) -> Any:
    """Instantiate a class from a class object, dotted import path, or file path."""
    if isinstance(cls, type):
        return cls(*args, **kwargs)

    if not isinstance(cls, str):
        raise TypeError(f"Class path must be a string. Got {type(cls)}")

    if ":" in cls:
        class_obj = resolve_class(cls)
        return class_obj(*args, **kwargs)

    instantiate_kwargs = dict(kwargs)
    if args:
        instantiate_kwargs["_args_"] = list(args)
    return instantiate({"_target_": cls, **instantiate_kwargs})


def _extract_param_help_from_docstring(docstring: str) -> dict[str, str]:
    """Parse NumPy-style ``Parameters`` doc blocks into ``name -> help`` text."""
    if not docstring:
        return {}

    lines = docstring.splitlines()
    in_params = False
    current_name = None
    current_desc: list[str] = []
    help_map: dict[str, str] = {}

    def _flush_current():
        nonlocal current_name, current_desc
        if current_name:
            description = " ".join(
                part.strip() for part in current_desc if part.strip()
            )
            if description:
                help_map[current_name] = description
        current_name = None
        current_desc = []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if not in_params:
            if stripped == "Parameters":
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if set(next_line) == {"-"} and len(next_line) >= 3:
                    in_params = True
                    i += 2
                    continue
        else:
            if stripped in {
                "Returns",
                "Raises",
                "Notes",
                "Examples",
                "See Also",
            }:
                _flush_current()
                break

            # Parameter declaration line: "name : type" or "name: type"
            if stripped and not lines[i].startswith((" ", "\t")) and (":" in stripped):
                _flush_current()
                current_name = stripped.split(":", 1)[0].strip()
            elif current_name is not None:
                current_desc.append(stripped)

        i += 1

    _flush_current()
    return help_map


def create_parser_from_function(
    func: Any,
    parser: Optional[argparse.ArgumentParser] = None,
    exclude: Optional[list] = None,
    **kwargs: Any,
) -> argparse.ArgumentParser:
    """
    Creates an argparse.ArgumentParser from a function's signature.

    Parameters
    ----------
    func: callable
        The function to create the parser from.
    parser : argparse.ArgumentParser, optional
        An existing parser to add arguments to. If None, a new parser is created.
    exclude: list, optional
        List of parameter names to exclude from the parser.
    **kwargs
        Additional keyword arguments to pass to the ArgumentParser constructor if a new parser is created.

    Raises
    ------
    ValueError
        If func is not callable or if parser is not an instance of argparse.ArgumentParser.


    Returns
    -------
    argparse.ArgumentParser
        The updated parser with arguments corresponding to the function's signature.
    """
    if not callable(func):
        raise ValueError(f"func must be callable. Got {type(func)}")

    if exclude is None:
        exclude = []

    docstring = inspect.getdoc(func)
    parser_description = None
    param_help = {}
    if docstring:
        parser_description = docstring.split("\n\n", 1)[0].strip()
        param_help = _extract_param_help_from_docstring(docstring)

    # Validate the parser
    conflict_handler = kwargs.pop("conflict_handler", "resolve")
    add_help = kwargs.pop("add_help", False)
    formatter_class = kwargs.pop(
        "formatter_class",
        argparse.RawDescriptionHelpFormatter,
    )
    if parser is None:
        parser = argparse.ArgumentParser(
            **kwargs,
            conflict_handler=conflict_handler,
            add_help=add_help,
            description=parser_description,
            formatter_class=formatter_class,
        )
    else:
        if len(kwargs) > 0:
            raise ValueError("Cannot pass kwargs when parser is provided.")
        if not isinstance(parser, argparse.ArgumentParser):
            raise ValueError(
                f"parser must be an instance of argparse.ArgumentParser or None. Got {type(parser)}",
            )
        if parser.description in [None, ""] and parser_description:
            parser.description = parser_description
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if name == "self" or name in exclude:
            continue
        if param.annotation is not inspect._empty:
            arg_type = param.annotation
        else:
            arg_type = str  # Default to string if no annotation
        help_text = param_help.get(name)
        if param.default is inspect._empty:
            parser.add_argument(
                f"--{name}",
                type=arg_type,
                required=True,
                help=help_text,
            )
        else:
            parser.add_argument(
                f"--{name}",
                type=arg_type,
                default=param.default,
                help=help_text,
            )
    return parser
