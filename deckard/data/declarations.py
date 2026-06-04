"""Unified dataset discovery declarations.

This module centralizes dataset discovery metadata across providers while
remaining safe when optional dependencies are not installed.

Two layers are exposed:

- ``discover_dataset_declarations``: provider-wide metadata discovery for docs,
  UX, and runtime introspection.
- ``build_loader_registry``: concrete ``dataset_name -> callable`` mappings for
  ``DataConfig`` loading (core + lifelines + yellowbrick).
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import pandas as pd
from sklearn.datasets import fetch_openml, make_classification, make_regression
from ..types import StringifiedClass

logger = logging.getLogger(__name__)


_ADULT_REPO_PATH_CANDIDATES = (
    Path("raw_data/adult_income/adult_income_dataset.csv"),
    Path("raw_data/adult_income/adult.csv"),
    Path("raw_data/adult_income/adult_income.csv"),
)

if TYPE_CHECKING:  # pragma: no cover
    from .base import DataConfig


@dataclass(frozen=True)
class DatasetDeclaration:
    """Declarative metadata for one discoverable dataset entry."""

    name: str
    provider: str
    target: str
    optional_dependency: str | None = None
    aliases: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"help": "Configuration field: aliases."},
    )
    notes: str = ""


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _discover_load_functions(module_name: str) -> list[tuple[str, str]]:
    """Return ``(dataset_name, target)`` pairs for ``load_*`` callables."""
    if not _module_available(module_name):
        return []
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug("Could not import %s for discovery: %s", module_name, exc)
        return []

    entries: list[tuple[str, str]] = []
    for attr in dir(module):
        if not attr.startswith("load_"):
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            entries.append((attr.replace("load_", "", 1), f"{module_name}.{attr}"))
    return entries


def _discover_huggingface_dataset_ids(
    limit: int = 100,
) -> list[tuple[str, int | None]]:
    """Return top Hugging Face dataset IDs with optional download counts.

    Discovery is best-effort and returns an empty list when the optional
    dependency is unavailable or network/API access fails.
    """
    if not _module_available("huggingface_hub"):
        return []
    try:
        from huggingface_hub import list_datasets

        # Support both newer and older huggingface_hub signatures.
        try:
            datasets = list_datasets(sort="downloads", direction=-1, limit=limit)
        except TypeError:
            datasets = list_datasets(sort="downloads", limit=limit)
        entries: list[tuple[str, int | None]] = []
        for item in datasets:
            dataset_id = getattr(item, "id", None)
            if not dataset_id:
                continue
            downloads = getattr(item, "downloads", None)
            entries.append((str(dataset_id), downloads))
        return entries
    except Exception as exc:
        logger.debug("HuggingFace dataset discovery failed: %s", exc)
        return []


def discover_provider_dataset_loaders(provider: str) -> dict[str, Callable[..., Any]]:
    """Discover ``load_*`` dataset callables for a provider.

    Args:
        provider: One of ``"lifelines"`` or ``"yellowbrick"``.
    """
    provider_map = {
        "lifelines": "lifelines.datasets",
        "yellowbrick": "yellowbrick.datasets",
    }
    if provider not in provider_map:
        raise ValueError(f"Unsupported provider '{provider}'")

    module_name = provider_map[provider]
    if not _module_available(module_name):
        return {}

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        logger.debug("Could not import %s for discovery: %s", module_name, exc)
        return {}

    dataset_map: dict[str, Callable[..., Any]] = {}
    for attr in dir(module):
        if not attr.startswith("load_"):
            continue
        loader = getattr(module, attr, None)
        if callable(loader):
            dataset_name = attr.replace("load_", "", 1)
            dataset_map[dataset_name] = loader
    return dataset_map


def discover_dataset_declarations() -> dict[str, DatasetDeclaration]:
    """Discover dataset declarations across core and optional providers.

    Returns a flat mapping keyed by dataset token.
    """
    declarations: dict[str, DatasetDeclaration] = {}

    # sklearn / core datasets handled by DataConfig.
    declarations.update(
        {
            "diabetes": DatasetDeclaration(
                name="diabetes",
                provider="sklearn",
                target="sklearn.datasets.load_diabetes",
                aliases=("sklearn.diabetes", "sklearn_diabetes"),
            ),
            "digits": DatasetDeclaration(
                name="digits",
                provider="sklearn",
                target="sklearn.datasets.load_digits",
                aliases=("sklearn.digits", "sklearn_digits"),
            ),
            "iris": DatasetDeclaration(
                name="iris",
                provider="sklearn",
                target="sklearn.datasets.load_iris",
                aliases=("sklearn.iris", "sklearn_iris"),
            ),
            "make_classification": DatasetDeclaration(
                name="make_classification",
                provider="sklearn",
                target="sklearn.datasets.make_classification",
                aliases=(
                    "sklearn.make_classification",
                    "sklearn_make_classification",
                ),
            ),
            "make_regression": DatasetDeclaration(
                name="make_regression",
                provider="sklearn",
                target="sklearn.datasets.make_regression",
                aliases=(
                    "sklearn.make_regression",
                    "sklearn_make_regression",
                ),
            ),
        },
    )

    # Lifelines datasets (optional, discovered from load_* functions).
    for dataset_name, target in _discover_load_functions("lifelines.datasets"):
        declarations.setdefault(
            dataset_name,
            DatasetDeclaration(
                name=dataset_name,
                provider="lifelines",
                target=target,
                optional_dependency="lifelines",
                aliases=(
                    f"lifelines_{dataset_name}",
                    f"lifelines.{dataset_name}",
                ),
            ),
        )

    # Yellowbrick datasets (optional, discovered from load_* functions).
    for dataset_name, target in _discover_load_functions("yellowbrick.datasets"):
        declarations.setdefault(
            dataset_name,
            DatasetDeclaration(
                name=dataset_name,
                provider="yellowbrick",
                target=target,
                optional_dependency="yellowbrick",
                aliases=(
                    f"yellowbrick_{dataset_name}",
                    f"yellowbrick.{dataset_name}",
                ),
            ),
        )

    # torchvision (optional): discover dataset classes if available.
    if _module_available("torchvision.datasets"):
        try:
            tv_ds = importlib.import_module("torchvision.datasets")
            for attr in dir(tv_ds):
                if attr.startswith("_"):
                    continue
                obj = getattr(tv_ds, attr, None)
                if isinstance(obj, type):
                    fqcn = f"torchvision.datasets.{attr}"
                    dataset_name = f"torchvision.{attr}"
                    declarations.setdefault(
                        dataset_name,
                        DatasetDeclaration(
                            name=dataset_name,
                            provider="torchvision",
                            target=fqcn,
                            optional_dependency="torch",
                            aliases=(fqcn, f"torchvision_{attr}"),
                        ),
                    )
        except Exception as exc:
            logger.debug("Torchvision dataset discovery failed: %s", exc)

    # Fairlearn-aware local datasets (optional torch path).
    declarations.setdefault(
        "fairlearn.TinyFairness",
        DatasetDeclaration(
            name="fairlearn.TinyFairness",
            provider="fairlearn",
            target="deckard.frameworks.pytorch.fairness_data.TinyFairness",
            optional_dependency="torch",
            aliases=(
                "fairlearn_TinyFairness",
                "deckard.frameworks.pytorch.fairness_data.TinyFairness",
            ),
            notes="Returns fairness-aware synthetic torch dataset.",
        ),
    )
    declarations.setdefault(
        "fairlearn.SyntheticImageSensitiveDataset",
        DatasetDeclaration(
            name="fairlearn.SyntheticImageSensitiveDataset",
            provider="fairlearn",
            target="deckard.frameworks.pytorch.fairness_data.SyntheticImageSensitiveDataset",
            optional_dependency="torch",
            aliases=(
                "fairlearn_SyntheticImageSensitiveDataset",
                "deckard.frameworks.pytorch.fairness_data.SyntheticImageSensitiveDataset",
            ),
        ),
    )

    # ART datasets (optional): discovered from art.utils load_* helpers.
    for dataset_name, target in _discover_load_functions("art.utils"):
        declarations.setdefault(
            f"art.{dataset_name}",
            DatasetDeclaration(
                name=f"art.{dataset_name}",
                provider="art",
                target=target,
                optional_dependency="art",
            ),
        )

    # HuggingFace datasets (optional): discover popular IDs from hub metadata.
    hf_entries = _discover_huggingface_dataset_ids(limit=100)
    if hf_entries:
        for dataset_id, downloads in hf_entries:
            canonical_name = f"huggingface.{dataset_id}"
            alias_name = f"huggingface_{dataset_id.replace('/', '_')}"
            notes = "Discovered from huggingface_hub.list_datasets(sort='downloads')."
            if downloads is not None:
                notes = f"{notes} downloads={downloads}"
            declarations.setdefault(
                canonical_name,
                DatasetDeclaration(
                    name=canonical_name,
                    provider="huggingface",
                    target="datasets.load_dataset",
                    optional_dependency="datasets",
                    aliases=(dataset_id, alias_name),
                    notes=notes,
                ),
            )
    elif _module_available("datasets"):
        # Fallback declaration keeps generic HuggingFace support discoverable.
        declarations.setdefault(
            "huggingface.dataset",
            DatasetDeclaration(
                name="huggingface.dataset",
                provider="huggingface",
                target="datasets.load_dataset",
                optional_dependency="datasets",
                aliases=("huggingface", "huggingface_dataset"),
                notes=(
                    "Use dataset_name like 'huggingface:imdb' or pass HF dataset "
                    "identifier via data_params when using custom loaders."
                ),
            ),
        )

    return declarations


def _repo_root_for_data_declarations() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_adult_copy_path(explicit_path: str | None = None) -> Path | None:
    repo_root = _repo_root_for_data_declarations()

    if explicit_path not in [None, ""]:
        candidate = Path(str(explicit_path)).expanduser()
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.exists():
            return candidate

    for relative_path in _ADULT_REPO_PATH_CANDIDATES:
        candidate = repo_root / relative_path
        if candidate.exists():
            return candidate

    return None


def load_adult_income_data(cfg: Any, **loader_params: Any) -> Any:
    """Load and preprocess Adult Income data into ``cfg._X``/``cfg._y``."""
    start_time = time.process_time()
    _ = loader_params
    adult = pd.read_csv(
        "https://raw.githubusercontent.com/simplymathematics/Adult-Census-Income/refs/heads/master/adult.csv",
        header=0,
    )
    print("*" * 80)
    print(adult.columns)
    print("*" * 80)
    y_raw = adult["income"]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
    else:
        y = cfg._encode_binary_series(
            y_raw.astype(str),
            {"<=50K": 0, ">50K": 1},
        )
    X = adult
    if "income" in adult.columns:
        del X["income"]
    if "sex" not in X.columns:
        raise ValueError("Adult dataset must include a 'sex' column")
    sex = cfg._encode_binary_series(
        X.pop("sex").astype(str),
        {"Male": 0, "Female": 1},
    )
    for column in [
        "age",
        "education.num",
        "hours.per.week",
        "capital-gain",
        "capital-loss",
        "fnlwgt",
    ]:
        if column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    categorical_columns = X.select_dtypes(
        include=["object", "category"],
    ).columns.tolist()
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True,
        dummy_na=True,
        dtype=int,
    )
    X["sex"] = sex.astype(int)
    cfg.data_load_time = time.process_time() - start_time
    cfg._X = X.apply(pd.to_numeric, errors="coerce")
    cfg._y = pd.Series(y)
    return cfg


def make_classification_data(
    cfg: Any,
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    n_redundant: int = 5,
    n_clusters_per_class: int = 2,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """Generate synthetic classification data into ``cfg._X``/``cfg._y``."""
    start_time = time.process_time()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        random_state=random_state,
        **kwargs,
    )
    cfg._X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    cfg._y = pd.Series(y)
    cfg.data_load_time = time.process_time() - start_time
    return cfg


def make_regression_data(
    cfg: Any,
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 10,
    noise: float = 0.1,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """Generate synthetic regression data into ``cfg._X``/``cfg._y``."""
    start_time = time.process_time()
    result = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
        **kwargs,
    )
    if isinstance(result, tuple) and len(result) >= 2:
        X, y = result[0], result[1]
    else:
        raise TypeError("make_regression did not return (X, y)")
    cfg._X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    cfg._y = pd.Series(y)
    cfg.data_load_time = time.process_time() - start_time
    return cfg


def load_generic_openml(
    cfg: Any,
    dataset_name: StringifiedClass,
    version: int = 1,
    **loader_params: Any,
) -> Any:
    """Load a generic OpenML dataset into ``cfg._X``/``cfg._y``."""
    start_time = time.process_time()
    dataset = fetch_openml(
        name=dataset_name,
        version=version,
        as_frame=True,
        **loader_params,
    )
    cfg._X = pd.DataFrame(dataset.data)
    cfg._y = pd.Series(dataset.target)
    cfg.data_load_time = time.process_time() - start_time
    return cfg


def load_generic_sklearn_dataset(
    cfg: Any,
    loader_name: str,
    **loader_params: Any,
) -> Any:
    """Load a sklearn bundled dataset by loader function name."""
    start_time = time.process_time()
    sklearn_datasets = importlib.import_module("sklearn.datasets")
    loader = getattr(sklearn_datasets, loader_name)
    dataset = loader(**loader_params)
    cfg._X = pd.DataFrame(dataset.data)
    cfg._y = pd.Series(dataset.target)
    cfg.data_load_time = time.process_time() - start_time
    return cfg


def load_lifelines_dataset(
    cfg: Any,
    dataset_name: StringifiedClass,
    **loader_params: Any,
) -> Any:
    """Load a lifelines dataset into ``cfg._X``/``cfg._y``."""
    lifelines_datasets = discover_provider_dataset_loaders("lifelines")
    if not lifelines_datasets:
        raise ImportError(
            "Lifelines datasets require optional dependency deckard[lifelines]",
        )
    if dataset_name not in lifelines_datasets:
        raise NotImplementedError(
            f"Lifelines dataset {dataset_name} not found. Supported: {sorted(lifelines_datasets.keys())}",
        )

    start_time = time.process_time()
    loader = lifelines_datasets[dataset_name]
    dataset = loader(**loader_params)
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)

    candidate_target = cfg.target
    if candidate_target is None:
        for candidate in ["E", "event", "status", "status_group"]:
            if candidate in dataset.columns:
                candidate_target = candidate
                break
    if candidate_target is None or candidate_target not in dataset.columns:
        candidate_target = "event"
        dataset[candidate_target] = 0

    y = dataset.pop(candidate_target)
    cfg.data_load_time = time.process_time() - start_time
    cfg._X = dataset
    cfg._y = pd.Series(y)
    return cfg


def load_yellowbrick_dataset(
    cfg: Any,
    dataset_name: StringifiedClass,
    **loader_params: Any,
) -> Any:
    """Load a yellowbrick dataset into ``cfg._X``/``cfg._y``."""
    yellowbrick_datasets = discover_provider_dataset_loaders("yellowbrick")
    if not yellowbrick_datasets:
        raise ImportError(
            "Yellowbrick datasets require optional dependency deckard[yellowbrick]",
        )
    if dataset_name not in yellowbrick_datasets:
        raise NotImplementedError(
            f"Yellowbrick dataset {dataset_name} not found. Supported: {sorted(yellowbrick_datasets.keys())}",
        )

    start_time = time.process_time()
    loader = yellowbrick_datasets[dataset_name]
    dataset = loader(**loader_params)

    if hasattr(dataset, "to_data") and callable(getattr(dataset, "to_data")):
        dataset = dataset.to_data()

    if isinstance(dataset, tuple) and len(dataset) == 2:
        X, y = dataset
    elif isinstance(dataset, pd.DataFrame):
        candidate_target = cfg.target
        if candidate_target is None:
            for candidate in ["target", "y", "label", "class"]:
                if candidate in dataset.columns:
                    candidate_target = candidate
                    break
        if candidate_target is None or candidate_target not in dataset.columns:
            candidate_target = "target"
            dataset[candidate_target] = 0
        y = dataset.pop(candidate_target)
        X = dataset
    elif hasattr(dataset, "data") and hasattr(dataset, "target"):
        X = dataset.data
        y = dataset.target
    else:
        raise TypeError(
            f"Unsupported Yellowbrick dataset output type: {type(dataset)}",
        )

    cfg.data_load_time = time.process_time() - start_time
    cfg._X = pd.DataFrame(X)
    cfg._y = pd.Series(y)
    return cfg


def load_default_dataset(
    cfg: "DataConfig",
    dataset_name: StringifiedClass,
    **loader_params: Any,
) -> Any:
    """Public default dataset loader entry-point for ``DataConfig``."""
    registry = build_loader_registry(cfg)
    if dataset_name not in registry:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    registry[dataset_name](**loader_params)
    return cfg


def build_loader_registry(cfg: "DataConfig") -> dict[str, Callable[..., Any]]:
    """Build ``DataConfig``-compatible dataset loader mapping.

    This is the runtime registry used by ``DataConfig._load_data`` and contains
    only callables that can load into tabular ``_X``/``_y`` for that config.
    """
    supported_datasets: dict[str, Callable[..., Any]] = {}

    def _register_sklearn(name: str, loader: Callable[..., Any]) -> None:
        # Keep legacy bare names while enforcing explicit sklearn-prefixed aliases.
        supported_datasets.setdefault(name, loader)
        supported_datasets.setdefault(f"sklearn.{name}", loader)
        supported_datasets.setdefault(f"sklearn_{name}", loader)

    supported_datasets.setdefault(
        "openml.adult",
        lambda **params: load_adult_income_data(cfg, **params),
    )
    supported_datasets.setdefault(
        "openml_adult",
        lambda **params: load_adult_income_data(cfg, **params),
    )
    _register_sklearn(
        "make_classification",
        lambda **params: make_classification_data(
            cfg,
            **params,
        ),
    )
    _register_sklearn(
        "make_regression",
        lambda **params: make_regression_data(cfg, **params),
    )
    _register_sklearn(
        "diabetes",
        lambda **params: load_generic_sklearn_dataset(
            cfg,
            "load_diabetes",
            **params,
        ),
    )
    _register_sklearn(
        "digits",
        lambda **params: load_generic_sklearn_dataset(
            cfg,
            "load_digits",
            **params,
        ),
    )
    _register_sklearn(
        "iris",
        lambda **params: load_generic_sklearn_dataset(
            cfg,
            "load_iris",
            **params,
        ),
    )

    # Optional providers keep aliases in sync with docs and legacy usage.
    # Register direct provider-discovered aliases first so names like
    # ``lifelines_diabetes`` are available even when bare names collide with
    # core sklearn datasets.
    for dataset_name in discover_provider_dataset_loaders("lifelines"):
        supported_datasets.setdefault(
            dataset_name,
            lambda _name=dataset_name, **params: load_lifelines_dataset(
                cfg,
                _name,
                **params,
            ),
        )
        supported_datasets.setdefault(
            f"lifelines_{dataset_name}",
            lambda _name=dataset_name, **params: load_lifelines_dataset(
                cfg,
                _name,
                **params,
            ),
        )
        supported_datasets.setdefault(
            f"lifelines.{dataset_name}",
            lambda _name=dataset_name, **params: load_lifelines_dataset(
                cfg,
                _name,
                **params,
            ),
        )

    for dataset_name in discover_provider_dataset_loaders("yellowbrick"):
        supported_datasets.setdefault(
            dataset_name,
            lambda _name=dataset_name, **params: load_yellowbrick_dataset(
                cfg,
                _name,
                **params,
            ),
        )
        supported_datasets.setdefault(
            f"yellowbrick_{dataset_name}",
            lambda _name=dataset_name, **params: load_yellowbrick_dataset(
                cfg,
                _name,
                **params,
            ),
        )
        supported_datasets.setdefault(
            f"yellowbrick.{dataset_name}",
            lambda _name=dataset_name, **params: load_yellowbrick_dataset(
                cfg,
                _name,
                **params,
            ),
        )

    for decl in discover_dataset_declarations().values():
        if decl.provider == "lifelines":
            dataset_name = decl.name
            supported_datasets.setdefault(
                dataset_name,
                lambda _name=dataset_name, **params: load_lifelines_dataset(
                    cfg,
                    _name,
                    **params,
                ),
            )
            for alias in decl.aliases:
                supported_datasets.setdefault(
                    alias,
                    lambda _name=dataset_name, **params: load_lifelines_dataset(
                        cfg,
                        _name,
                        **params,
                    ),
                )

        if decl.provider == "yellowbrick":
            dataset_name = decl.name
            supported_datasets.setdefault(
                dataset_name,
                lambda _name=dataset_name, **params: load_yellowbrick_dataset(
                    cfg,
                    _name,
                    **params,
                ),
            )
            for alias in decl.aliases:
                supported_datasets.setdefault(
                    alias,
                    lambda _name=dataset_name, **params: load_yellowbrick_dataset(
                        cfg,
                        _name,
                        **params,
                    ),
                )

    return supported_datasets


__all__ = [
    "DatasetDeclaration",
    "discover_dataset_declarations",
    "build_loader_registry",
    "load_adult_income_data",
    "make_classification_data",
    "make_regression_data",
    "load_generic_sklearn_dataset",
    "load_generic_openml",
    "discover_provider_dataset_loaders",
    "load_lifelines_dataset",
    "load_yellowbrick_dataset",
    "load_default_dataset",
]
