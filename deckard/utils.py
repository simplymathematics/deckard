"""Shared utility layer for deckard's public Python API.

This module contains the base configuration protocol used across the project,
stable hashing helpers for config identity, file IO helpers, and utility
functions for dynamically resolving and instantiating classes.
"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from .artifacts import ArtifactLoaderConfig, ScoreDict

logger = logging.getLogger(__name__)

__all__ = [
    "BaseConfig",
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
    "split_separated_tokens",
    "normalize_hydra_list_overrides",
    "merge_list_of_dicts",
    "merge_scores_with_collision_suffix",
    "resolve_torch_device",
    "probabilities_from_model_outputs",
]


NULL_CONFIG_TOKENS = frozenset({"none", "null", "n/a"})
DEFAULT_CONFIG_TOKENS = frozenset({"auto", "default", "best"})

RuntimeScalar = str | int | float | bool | None
RuntimeSerializable = (
    RuntimeScalar
    | list["RuntimeSerializable"]
    | tuple["RuntimeSerializable", ...]
    | dict[str, "RuntimeSerializable"]
)
ComponentInput = "BaseConfig | DictConfig | ListConfig | RuntimeSerializable"


def normalize_config_token(value: Any) -> str | None:
    """
    Normalize a config token to lower-case text for keyword matching.

    Args:
        value (Any): The value to normalize.

    Returns:
        Optional[str]: Normalized string or None if input is None.
    """
    if value is None:
        return None
    return str(value).strip().lower()


def is_null_config_value(value: Any, *, allow_empty: bool = True) -> bool:
    """
    Return True when value represents an explicit null-like config token.

    Args:
        value (Any): Value to check.
        allow_empty (bool, optional): Whether to treat empty string as null. Defaults to True.

    Returns:
        bool: True if value is null-like, False otherwise.
    """
    token = normalize_config_token(value)
    if token is None:
        return True
    if allow_empty and token == "":
        return True
    return token in NULL_CONFIG_TOKENS


def is_default_config_value(value: Any, *, include_best: bool = True) -> bool:
    """
    Return True when value requests default/auto config behavior.

    Args:
        value (Any): Value to check.
        include_best (bool, optional): Whether to include 'best' as default. Defaults to True.

    Returns:
        bool: True if value is a default/auto config token, False otherwise.
    """
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
    """
    Resolve the best torch device given a user/device request.

    Args:
        requested_device (Any, optional): Device specifier (int, str, torch.device, or None).

    Returns:
        Any: torch.device or string representing the resolved device.

    Example:
        >>> resolve_torch_device('cuda:0')
        device(type='cuda', index=0)
    """
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
    """
    Register a Hydra config node while tolerating duplicate registrations.

    Args:
        group (str): Hydra config group.
        name (str): Config name.
        node (Any): Config node to register.
    """
    cs = ConfigStore.instance()
    try:
        cs.store(group=group, name=name, node=node)
    except Exception:
        # Re-imports in tests/dev can register the same node repeatedly.
        pass


def coerce_to_list(items: Union[list, Any]) -> list:
    """
    Normalize a list or OmegaConf ListConfig to a plain Python list.

    Args:
        items (Union[list, Any]): A list or OmegaConf ListConfig.

    Returns:
        list: A plain Python list whose elements are the same objects as items.

    Raises:
        TypeError: If items is neither a list nor a ListConfig.
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

    Args:
        value (Any): The value to normalize. Accepts ``None``, null-like
            tokens, ``str``, ``list``, or ``ListConfig``.
        field_name (str): Name of the field being normalized, used in error
            messages. Defaults to ``"value"``.

    Returns:
        Optional[list]: A plain Python list, or ``None`` when *value* is
            null-like.

    Raises:
        TypeError: If *value* is not a recognized type.
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

    Args:
        value (Any): The value to normalize. Accepts ``None``, null-like
            tokens, ``bool``, ``dict``, ``DictConfig``, ``list``, or
            ``ListConfig``.
        field_name (str): Name of the field being normalized, used in error
            messages.

    Returns:
        Union[None, bool, dict]: ``None`` for null-like tokens, the original
        ``bool`` unchanged, or a plain ``dict`` merged from mapping/list input.

    Raises:
        TypeError: If *value* is not a recognized type.

    Note:
        ``list`` / ``ListConfig`` values are merged via
        {func}`deckard.utils.merge_list_of_dicts`.
    """
    if isinstance(value, (list, ListConfig)):
        return merge_list_of_dicts(coerce_to_list(value))
    if isinstance(value, bool):
        return value
    if is_null_config_value(value):
        return None
    if isinstance(value, DictConfig):
        return {str(key): item for key, item in dict(value).items()}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    raise TypeError(
        f"{field_name} must be None, bool, dict/DictConfig, or list/ListConfig. Got {type(value)}",
    )


def normalize_plugin_specs(plugins: Any) -> list:
    """Normalize plugin config values into a plain list.

    Args:
        plugins (Any): Plugin config value to normalize. Accepts ``None``,
            null-like tokens, ``list``, and OmegaConf ``ListConfig``. Bare
            non-list values (e.g. a plain string or dict) are rejected.

    Returns:
        list: A plain Python list of plugin specs (empty when *plugins* is
            null-like).

    Raises:
        TypeError: If *plugins* is not ``None``, a list, or a ``ListConfig``.
    """
    if is_null_config_value(plugins):
        return []
    if isinstance(plugins, (ListConfig, list)):
        return list(plugins)
    raise TypeError(
        f"plugins must be a list or None, got {type(plugins).__name__!r}: {plugins!r}",
    )


def instantiate_plugin_spec(
    plugin_spec: Any,
    *,
    loader,
) -> Any:
    """Instantiate one plugin spec using a caller-provided loader.

    Args:
        plugin_spec (Any): A ``dict`` with ``name`` or ``_target_`` key, a
            dotted import-path string, a class type, or an already-instantiated
            object.
        loader (callable): Callable with signature ``loader(path, **kwargs)``
            used to resolve string/dict specs into instances.

    Returns:
        Any: The instantiated plugin object.

    Raises:
        ValueError: If *plugin_spec* is a dict without a ``name`` or
            ``_target_`` key.
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


def split_separated_tokens(value: Any, sep=",") -> list[str]:
    """Split a comma-separated string into trimmed, non-empty tokens.

    Args:
        value (Any): Value to split. Converted to ``str`` before splitting.
            Returns an empty list when *value* is ``None``.

    Returns:
        list[str]: List of whitespace-stripped, non-empty token strings.
    """
    if value is None:
        return []
    return [item.strip() for item in str(value).split(sep) if item.strip() != ""]


def normalize_hydra_list_overrides(
    overrides: list[str],
    *,
    keys: tuple[str, ...] = ("score",),
) -> list[str]:
    """Rewrite comma-separated [Hydra](https://hydra.cc) override values into list syntax.

    Args:
        overrides (list[str]): Raw Hydra override tokens from the CLI or
            programmatic composition.
        keys (tuple[str, ...]): Override keys whose comma-separated values
            should be rewritten. Defaults to ``("score",)``.

    Returns:
        list[str]: Rewritten override list where matching keys use bracketed
        list syntax (e.g. ``score=[a,b]``). Non-matching tokens are passed
        through unchanged.

    Example:
        ```python
        normalize_hydra_list_overrides(["score=a,b"])
        # ["score=[a,b]"]
        ```
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

        items = split_separated_tokens(stripped)
        normalized.append(f"{key}=[{','.join(items)}]")

    return normalized


def merge_list_of_dicts(items: Iterable) -> dict:
    """Merge a sequence of dict-like items into a single ``dict``.

    Each element of *items* may be a plain ``dict`` or an OmegaConf
    ``DictConfig``; the latter is converted via ``OmegaConf.to_container``
    before merging.  When two elements share a key, the later element wins.

    Args:
        items: Iterable of ``dict`` or ``DictConfig`` objects.

    Returns:
        Single merged plain-Python ``dict``.

    Raises:
        TypeError: If any element cannot be resolved to a ``dict``.
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

    Args:
        base_scores (dict): Existing scores to merge into.
        incoming_scores (dict): New scores to add. Colliding keys are
            disambiguated.
        alias (Optional[str]): When provided, colliding keys are suffixed with
            ``_<alias>`` (and ``_<alias>_2``, etc. for further collisions).
            When ``None``, incoming values overwrite existing keys without
            suffixing.

    Returns:
        dict: Merged score dictionary.

    Raises:
        TypeError: If *base_scores* or *incoming_scores* is not a ``dict``.
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
    - {class}`deckard.utils.ConfigBase` -> ``dict`` via ``to_dict``
    - existing YAML file path string -> loaded config container via OmegaConf

    Args:
        config_obj (Any): Object to coerce. Returned unchanged when none of
            the supported coercions apply.

    Returns:
        Any: A plain Python ``dict`` / ``list``, or the original object if no
        coercion was applicable.
    """
    if config_obj is None:
        return None

    if isinstance(config_obj, DictConfig):
        return OmegaConf.to_container(config_obj, resolve=True)

    if isinstance(config_obj, BaseConfig):
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
    """Normalize config-like input into a [Hydra](https://hydra.cc) instantiation dictionary.

    When ``default_target`` is provided and the resulting mapping lacks a
    ``_target_`` key, the target is injected so Hydra can instantiate the
    canonical config class while still allowing user-specified overrides.

    Args:
        config_obj (Any): Config-like input coerced via
            {func}`deckard.utils.coerce_config` before dict construction.
        default_target (Optional[str]): Dotted import path injected as
            ``_target_`` when the resolved mapping does not already contain
            one.

    Returns:
        dict[str, Any]: Instantiation dictionary suitable for ``hydra.utils.instantiate``.

    Raises:
        TypeError: If *config_obj* does not resolve to a ``dict``.
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
    """Instantiate one config object via [Hydra](https://hydra.cc), injecting a default target when needed.

    Args:
        config_obj (Any): Config-like input to instantiate. Returned unchanged
            when already an instance of *expected_type*. ``None`` returns
            ``None``.
        expected_type (type): Required type of the instantiated result.
        default_target (Optional[str]): Dotted import path used as ``_target_``
            when not already present. Defaults to the fully-qualified name of
            *expected_type*.

    Returns:
        Any: Instantiated config object of type *expected_type*, or ``None``.

    Raises:
        TypeError: If the instantiated object is not an instance of
            *expected_type*.
    """
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

    Args:
        scores (dict): Score dictionary with numeric or non-numeric values.
        n_samples (int): Number of samples used to determine rounding
            precision. Values ``<= 0`` or ``None`` fall back to one decimal
            place.
        logger_obj (Optional[logging.Logger]): When provided, each rounded
            value is logged at INFO level. Defaults to ``None``.

    Returns:
        dict: New score dictionary with numeric values rounded in-place.
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

    Accepts logits or probabilities from [PyTorch](https://pytorch.org) or
    [NumPy](https://numpy.org) outputs and applies softmax (multi-class) or
    sigmoid (binary) normalization as needed.

    Args:
        outputs (Any): Raw model outputs — a PyTorch tensor or NumPy array of
            shape ``(n_samples,)`` or ``(n_samples, n_classes)``.

    Returns:
        np.ndarray: Array of shape ``(n_samples, n_classes)`` with
        probabilities summing to 1 across the class axis.

    Raises:
        ValueError: If *outputs* has an unrecognized shape.
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


@dataclass(init=False)
class BaseConfig(ArtifactLoaderConfig):
    """Base class for deckard configuration objects.

    ``ConfigBase`` provides a common lifecycle for config dataclasses: argument
    hydration, post-init hooks, stable hashing based on configuration state, and
    serialization helpers used throughout deckard.
    """

    # _target_: str = "deckard.utils.ConfigBase"
    score_dict: ScoreDict = field(default_factory=ScoreDict)
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

    def __post_init__(self):
        pass

    def _before_post_init(self) -> None:
        """Hook for subclasses that need pre-normalization before __post_init__."""

    def _after_post_init(self) -> None:
        """Finalize common lifecycle state after __post_init__."""
        self.score_dict = ScoreDict.from_payload(getattr(self, "score_dict", {}))
        # Freeze hash at configuration time so runtime attributes added during
        # execution cannot alter experiment identity.
        self._hash_payload = self.to_dict(for_hash=True)
        self._hash_value = hash_conf_values(self._hash_payload)

    def coerce_component(
        self,
        component: ComponentInput,
        expected_type: type,
        *,
        default_target: Optional[str] = None,
        overrides: Optional[dict[str, RuntimeSerializable]] = None,
        allow_passthrough: Callable[[ComponentInput], bool] | None = None,
    ) -> "BaseConfig | DictConfig | ListConfig | RuntimeSerializable | None":
        """Instantiate/normalize a config component to ``expected_type``.

        Args:
            component: Config-like object, runtime object, or expected config instance.
            expected_type: Required config class.
            default_target: Hydra target path used when input lacks ``_target_``.
            overrides: Key/value pairs injected into the normalized spec.
            allow_passthrough: Optional predicate that bypasses coercion.

        Returns:
            Instantiated config, passthrough runtime payload, or None.

        Raises:
            TypeError: If instantiated payload is not an instance of expected_type.
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

    def __call__(self) -> ScoreDict:
        """Execute runtime behavior and return normalized score payload.

        Returns:
            Normalized score payload for the runtime execution.

        Raises:
            NotImplementedError: Always raised by the abstract base implementation.
        """
        raise NotImplementedError("This is an abstract base class.")

    def __hash__(self):
        """Return the initialization-time configuration hash as int."""
        if "_hash_value" not in self.__dict__:
            self._hash_payload = self.to_dict(for_hash=True)
            self._hash_value = hash_conf_values(self._hash_payload)
        return int(self._hash_value, 16)

    def __eq__(self, other: object) -> bool:
        """Two ConfigBase instances are equal when their configuration hashes match."""
        if not isinstance(other, BaseConfig):
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

    def read_or_initialize_scores(self, score_file: Optional[str]) -> dict:
        """Return merged scores from disk and memory, or initialize output location.

        Args:
            score_file: Optional score file path.

        Returns:
            Merged score payload dictionary persisted via score lifecycle hooks.
        """
        runtime_scores = ScoreDict.from_payload(getattr(self, "score_dict", {}))
        resolved = runtime_scores(
            score_file=score_file,
            artifact_loader=self,
            persist=True,
        )
        self.score_dict = ScoreDict.from_payload(resolved)
        return dict(self.score_dict)

    def merge_and_persist_scores(
        self,
        new_scores: dict,
        score_file: Optional[str],
    ) -> dict:
        """Merge score payload with on-disk scores and persist via ScoreDict lifecycle.

        Args:
            new_scores: Newly computed score payload.
            score_file: Optional score file path.

        Returns:
            Canonical merged score payload.
        """
        merged_input = ScoreDict.from_payload(new_scores)
        resolved = merged_input(
            score_file=score_file,
            artifact_loader=self,
            persist=True,
        )
        return dict(ScoreDict.from_payload(resolved))

    @staticmethod
    def _coerce_files_mapping(files_value: Any) -> dict[str, Any]:
        """Coerce supported file containers into a plain mapping."""
        if files_value is None:
            return {}
        if isinstance(files_value, dict):
            return dict(files_value)
        if hasattr(files_value, "as_dict") and callable(files_value.as_dict):
            candidate = files_value.as_dict()
            if isinstance(candidate, dict):
                return dict(candidate)
        return {}

    def merge_runtime_files(
        self,
        *file_mappings: RuntimeSerializable,
        include_existing: bool = True,
        update_score_dict: bool = True,
    ) -> dict[str, RuntimeSerializable]:
        """Merge runtime file mappings and persist merged map on the config.

        This is used at the end of runtime ``__call__`` paths, just before
        persistence, so newly created artifact paths are retained in-memory and
        optionally attached to ``score_dict`` for on-disk score metadata.

        Args:
            *file_mappings: File mapping containers to merge.
            include_existing: Whether to include current ``self.files`` entries.
            update_score_dict: Whether to mirror merged file map into ``score_dict``.

        Returns:
            Merged runtime file mapping.
        """
        merged: dict[str, RuntimeSerializable] = {}
        if include_existing:
            merged.update(self._coerce_files_mapping(getattr(self, "files", None)))
        for mapping in file_mappings:
            merged.update(self._coerce_files_mapping(mapping))

        if hasattr(self, "files"):
            files_attr = getattr(self, "files", None)
            if isinstance(files_attr, dict):
                setattr(self, "files", dict(merged))

        if update_score_dict and getattr(self, "score_dict", None) is not None:
            self.score_dict = ScoreDict.from_payload(getattr(self, "score_dict", {}))
            self.score_dict["files"] = dict(merged)
        return merged

    def get_call_params(self) -> dict:
        """Retrieve parameters required to call the instance ``__call__`` method.

        Returns:
            Mapping of ``__call__`` parameter names to current instance values.

        Raises:
            AttributeError: If a required ``__call__`` parameter is missing on the instance.
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

    @staticmethod
    def _resolve_yaml_write_path(filepath: str) -> Path:
        """Return canonical YAML output path for config state persistence."""
        path = Path(filepath)
        if path.suffix.lower() not in {".yaml", ".yml"}:
            path = path.with_suffix(".yaml")
        return path

    @staticmethod
    def _resolve_yaml_read_path(filepath: str) -> Path:
        """Resolve config state path, preferring canonical YAML suffixes."""
        path = Path(filepath)
        candidates = [path]
        if path.suffix.lower() not in {".yaml", ".yml"}:
            candidates.insert(0, path.with_suffix(".yaml"))
            candidates.insert(1, path.with_suffix(".yml"))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def from_yaml(filepath: str) -> "BaseConfig":
        """
        Creates an instance of the class from a YAML configuration file.

        Args:
            filepath: Path to YAML configuration file.

        Returns:
            Instance initialized from the YAML configuration.

        Raises:
            TypeError: If loaded YAML content does not deserialize to a dictionary.
        """
        resolved_path = BaseConfig._resolve_yaml_read_path(filepath)
        config = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
        if not isinstance(config, dict):
            raise TypeError(
                f"Loaded config is not a dictionary from {resolved_path}",
            )
        instance = instantiate(config)
        logger.info(
            f"Instance of {instance.__class__.__name__} created from {resolved_path}",
        )
        return instance

    @staticmethod
    def from_dict(data: dict) -> "BaseConfig":
        """
        Creates an instance of the class from a dictionary.

        Args:
            data: Dictionary containing the configuration.

        Returns:
            Instance initialized from dictionary configuration.
        """
        instance = instantiate(data)
        return instance

    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Convert the current instance to YAML and optionally persist to YAML file.

        Args:
            filepath: Optional destination path.

        Returns:
            YAML text when filepath is None, otherwise the persisted YAML path.
        """
        config = self.to_dict()
        config = OmegaConf.create(config)
        yaml_text = str(OmegaConf.to_yaml(config))
        if filepath is not None:
            path = self._resolve_yaml_write_path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_text, encoding="utf-8")
            return str(path)
        return yaml_text

    def to_dict(self, for_hash: bool = False) -> dict:
        """
        Converts the current instance to a dictionary.

        Args:
            for_hash: Whether to omit runtime-only fields for stable hashing.

        Returns:
            Dictionary representation of the instance.
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
                    if isinstance(value, BaseConfig):
                        dict_[name] = value.to_dict(for_hash=for_hash)
                    elif OmegaConf.is_config(value):
                        dict_[name] = OmegaConf.to_container(
                            value,
                            resolve=True,
                        )
                    else:
                        dict_[name] = self._serialize_for_yaml(value)

        # Include any additional runtime attrs not declared as dataclass fields
        for name, value in self.__dict__.items():
            if (
                name.startswith("_") and not (for_hash and name == "_target_")
            ) or name in dict_:
                continue
            if for_hash and not self._is_hash_field(name):
                continue
            if isinstance(value, BaseConfig):
                dict_[name] = value.to_dict(for_hash=for_hash)
            elif OmegaConf.is_config(value):
                dict_[name] = OmegaConf.to_container(value, resolve=True)
            else:
                dict_[name] = self._serialize_for_yaml(value)

        dict_["_target_"] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        return dict_

    @staticmethod
    def _serialize_for_yaml(value: Any) -> Any:
        """Convert non-primitive values into YAML-safe representations."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if is_dataclass(value) and not isinstance(value, type):
            return BaseConfig._serialize_for_yaml(asdict(value))
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        if callable(value):
            module = getattr(value, "__module__", None)
            qualname = getattr(value, "__qualname__", None)
            if module is not None and qualname is not None:
                return f"{module}.{qualname}"
            return str(value)
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, dict):
            return {
                str(k): BaseConfig._serialize_for_yaml(v) for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [BaseConfig._serialize_for_yaml(v) for v in value]
        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            try:
                data = value.to_dict()
                return BaseConfig._serialize_for_yaml(data)
            except Exception:
                return str(value)
        if hasattr(value, "__dict__"):
            public_attrs = {
                k: v
                for k, v in value.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }
            if public_attrs:
                return BaseConfig._serialize_for_yaml(public_attrs)
        return str(value)

    def execute_without_mercy(self) -> dict:
        """Execute config runtime and persist traceback details on failure.

        Returns:
            Score payload produced by runtime execution or fallback ``score_dict``.
        """
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


# Backward-compatible alias for legacy Hydra targets.
ConfigBase = BaseConfig

def save_data(
    data: pd.DataFrame,
    filepath: Union[str, None] = None,
    **kwargs,
) -> None:
    """Persist tabular data via ArtifactLoaderConfig IO contract."""
    loader = ArtifactLoaderConfig(payload_kind="data")
    loader.save_data(data, filepath, **kwargs)
    if filepath is not None:
        logger.info(f"Data saved to {Path(filepath)}")


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """Load tabular data via ArtifactLoaderConfig IO contract.

    Args:
        filepath: Source data file path.
        **kwargs: Extra keyword args forwarded to pandas readers.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If ``filepath`` is ``None``.
        ValueError: If the file extension is unsupported.
    """

    loader = ArtifactLoaderConfig(payload_kind="data")
    data = loader.load_data(filepath, **kwargs)
    logger.info(f"Data loaded from {Path(filepath)}")
    return pd.DataFrame(data)


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
    """Create an ``argparse.ArgumentParser`` from a function signature.

    Args:
        func: Callable used to derive CLI arguments.
        parser: Existing parser to extend. When ``None``, a new parser is created.
        exclude: Optional parameter names to skip.
        **kwargs: Extra parser-constructor kwargs when ``parser`` is ``None``.

    Returns:
        Parser populated with arguments derived from ``func``.

    Raises:
        ValueError: If ``func`` is not callable or parser input is invalid.
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
