from __future__ import annotations

# Standard library
import logging
from dataclasses import dataclass
from typing import Any

# Third-party
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torch import Tensor
from torch.utils.data import TensorDataset

# Local / project
from ...data.sample import BaseSampler
from ...utils import load_class

try:
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - optional dependency fallback
    DictConfig = None
    OmegaConf = None

# Logger
logger = logging.getLogger(__name__)

MatrixLike = Tensor
ArrayLike = Tensor

__all__ = [
    "PytorchBaseSampler",
    "PytorchSplitSampler",
    "PytorchFoldSampler",
    "PytorchShuffleSampler",
    "PytorchKFoldSampler",
    "TorchBaseSampler",
    "TorchSplitSampler",
    "TorchKFoldSampler",
    "TorchShuffleSampler",
]


@dataclass
class PytorchBaseSampler(BaseSampler):
    """PyTorch sampler base that mirrors :class:`deckard.data.sample.BaseSampler`.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def __call__(self, config: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run sampling strategy against runtime config.

        Args:
            config: Runtime data-like config.

        Returns:
            Train, test, and validation index arrays.

        Raises:
            NotImplementedError: Always raised by base sampler interface.
        """
        raise NotImplementedError

    @staticmethod
    def _cfg_value(
        config: Any,
        sampler_params: dict[str, Any],
        name: str,
        default: Any,
    ) -> Any:
        getter = getattr(config, "_get_sampler_option", None)
        if callable(getter):
            return getter(name, default)
        if name in sampler_params:
            return sampler_params[name]
        return default

    @classmethod
    def _sampler_kwargs_for_alias(
        cls,
        config: Any,
        sampler_params: dict[str, Any],
        alias: str,
    ) -> dict[str, Any]:
        if alias == "split":
            return {
                "train_size": cls._cfg_value(
                    config,
                    sampler_params,
                    "train_size",
                    None,
                ),
                "test_size": cls._cfg_value(config, sampler_params, "test_size", None),
                "val_size": cls._cfg_value(config, sampler_params, "val_size", None),
                "random_state": cls._cfg_value(
                    config,
                    sampler_params,
                    "random_state",
                    42,
                ),
                "stratify": cls._cfg_value(config, sampler_params, "stratify", True),
            }
        if alias in {"fold", "kfold"}:
            return {
                "n_splits": getattr(config, "n_splits", 5),
                "split": getattr(config, "split", 0),
                "train_size": cls._cfg_value(
                    config,
                    sampler_params,
                    "train_size",
                    None,
                ),
                "test_size": cls._cfg_value(config, sampler_params, "test_size", None),
                "val_size": cls._cfg_value(config, sampler_params, "val_size", None),
                "random_state": cls._cfg_value(
                    config,
                    sampler_params,
                    "random_state",
                    42,
                ),
                "stratify": cls._cfg_value(config, sampler_params, "stratify", True),
            }
        if alias == "shuffle":
            return {
                "n_splits": getattr(config, "n_splits", 5),
                "split": getattr(config, "split", 0),
                "test_size": cls._cfg_value(config, sampler_params, "test_size", None),
                "val_size": cls._cfg_value(config, sampler_params, "val_size", None),
                "random_state": cls._cfg_value(
                    config,
                    sampler_params,
                    "random_state",
                    42,
                ),
                "stratify": cls._cfg_value(config, sampler_params, "stratify", True),
            }
        return {}

    @classmethod
    def resolve(cls, config: Any) -> Any:
        """Resolve sampler declaration into callable sampler object.

        Args:
            config: Runtime data-like config.

        Returns:
            Resolved sampler object or ``None``.

        Raises:
            ValueError: If sampler declaration is invalid or unsupported.
        """
        sampler_aliases = {
            "split": PytorchSplitSampler,
            "fold": PytorchFoldSampler,
            "kfold": PytorchFoldSampler,
            "shuffle": PytorchShuffleSampler,
        }

        sampler_params = dict(getattr(config, "sampler_params", {}) or {})

        spec = getattr(config, "sampler", None)
        if spec is None:
            return None

        if isinstance(spec, str):
            key = spec.strip().lower()
            if key not in sampler_aliases:
                raise ValueError(
                    f"Unknown sampler '{spec}'. Must be one of {list(sampler_aliases)}.",
                )
            alias_kwargs = cls._sampler_kwargs_for_alias(config, sampler_params, key)
            alias_kwargs.update(sampler_params)
            return sampler_aliases[key](**alias_kwargs)

        if (
            DictConfig is not None
            and OmegaConf is not None
            and isinstance(spec, DictConfig)
        ):
            spec = OmegaConf.to_container(spec, resolve=True)

        if isinstance(spec, dict):
            if not spec:
                return None
            spec = dict(spec)
            class_path = spec.pop("name", spec.pop("_target_", None))
            if class_path is None:
                raise ValueError("sampler dict must include 'name' or '_target_'")

            key = str(class_path).strip().lower()
            if key in sampler_aliases:
                alias_kwargs = cls._sampler_kwargs_for_alias(
                    config,
                    sampler_params,
                    key,
                )
                alias_kwargs.update(sampler_params)
                alias_kwargs.update({str(k): v for k, v in spec.items()})
                return sampler_aliases[key](**alias_kwargs)

            loaded = load_class(class_path, **{str(k): v for k, v in spec.items()})
            if isinstance(loaded, type):
                return loaded()
            return loaded

        return cls._resolve_terminal_sampler_spec(spec)

    @classmethod
    def _default_sampler(cls) -> TorchBaseSampler:
        return PytorchSplitSampler()

    @classmethod
    def _configure_composed_sampler(cls, sampler_obj: Any, config: Any) -> Any:
        params = dict(getattr(config, "sampler_params", {}) or {})
        for field_name, value in params.items():
            if hasattr(sampler_obj, field_name):
                setattr(sampler_obj, field_name, value)
        return sampler_obj

    @staticmethod
    def _validate_stratify(stratify: bool | str | None) -> None:
        if stratify not in (None, True, False):
            raise ValueError(
                f"stratify must be None, True, or False for PyTorch samplers; got {stratify}.",
            )

    @staticmethod
    def _validate_fractional_sizes(
        train_size: int | float | None,
        test_size: int | float | None,
        val_size: int | float | None,
    ) -> None:
        sizes = {
            "train_size": train_size,
            "test_size": test_size,
            "val_size": val_size,
        }
        for name, value in sizes.items():
            if value is None:
                continue
            if isinstance(value, (int, np.integer)):
                if int(value) < 0:
                    raise ValueError(f"{name} must be >= 0, got {value}.")
                continue
            if isinstance(value, float):
                if value < 0.0 or value > 1.0:
                    raise ValueError(
                        f"{name} float value must be in [0, 1], got {value}.",
                    )
                continue
            raise TypeError(f"{name} must be int, float, or None, got {type(value)}.")

        all_fractional_or_none = all(
            value is None or isinstance(value, float) for value in sizes.values()
        )
        if all_fractional_or_none:
            total = sum(float(value or 0.0) for value in sizes.values())
            if total > 1.0 + 1e-8:
                raise ValueError(
                    "When using float sizes, train_size + test_size + val_size must be <= 1.0.",
                )


@dataclass
class PytorchSplitSampler(PytorchBaseSampler):
    """PytorchSplitSampler runtime class.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    train_size: int | float | None = None
    test_size: int | float | None = 0.2
    val_size: int | float | None = None
    random_state: int | None = 42
    stratify: bool | str | None = True

    def __call__(self, config: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate deterministic train/test/validation splits.

        Args:
            config: Runtime data-like config.

        Returns:
            Train, test, and validation index arrays.

        Raises:
            ValueError: If split sizes are invalid or source data is unavailable.
        """
        self._validate_stratify(self.stratify)
        self._validate_fractional_sizes(self.train_size, self.test_size, self.val_size)
        if self.train_size is None and self.test_size is None:
            raise ValueError("Either train_size or test_size must be specified.")
        dataset = getattr(config, "dataset_obj", None)
        if dataset is None:
            if (
                getattr(config, "_X", None) is None
                or getattr(config, "_y", None) is None
            ):
                raise ValueError("Data not loaded. Call load_dataset() first.")
            dataset = TensorDataset(config._X, config._y)
        indices = np.arange(len(dataset))
        labels = getattr(config, "_y", None)
        y = (
            labels.detach().cpu().numpy()
            if (self.stratify and labels is not None)
            else None
        )

        if self.val_size is not None and float(self.val_size) > 0:
            train_test_idx, val_idx = train_test_split(
                indices,
                test_size=self.val_size,
                random_state=self.random_state,
                stratify=y if y is not None else None,
            )
            y_sub = y[train_test_idx] if y is not None else None
            train_idx, test_idx = train_test_split(
                train_test_idx,
                train_size=self.train_size,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_sub,
            )
        else:
            train_idx, test_idx = train_test_split(
                indices,
                train_size=self.train_size,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if y is not None else None,
            )
            val_idx = np.array([], dtype=np.int64)

        return train_idx, test_idx, val_idx


@dataclass
class PytorchFoldSampler(PytorchBaseSampler):
    """PytorchFoldSampler runtime class.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    n_splits: int = 5
    split: int = 0
    shuffle: bool = True
    train_size: int | float | None = None
    test_size: int | float | None = 0.2
    val_size: int | float | None = None
    random_state: int | None = 42
    stratify: bool | str | None = True

    @staticmethod
    def _to_count(size, total: int):
        return BaseSampler._size_to_count(size, total)

    def __call__(self, config: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate fold-based train/test/validation splits.

        Args:
            config: Runtime data-like config.

        Returns:
            Train, test, and validation index arrays.

        Raises:
            ValueError: If split index or sizing constraints are invalid.
        """
        self._validate_stratify(self.stratify)
        self._validate_fractional_sizes(self.train_size, self.test_size, self.val_size)
        if self.n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {self.n_splits}.")
        dataset = getattr(config, "dataset_obj", None)
        if dataset is None:
            if (
                getattr(config, "_X", None) is None
                or getattr(config, "_y", None) is None
            ):
                raise ValueError("Data not loaded. Call load_dataset() first.")
            dataset = TensorDataset(config._X, config._y)
        indices = np.arange(len(dataset))
        labels = getattr(config, "_y", None)
        y = (
            labels.detach().cpu().numpy()
            if (self.stratify and labels is not None)
            else None
        )

        if y is not None:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state if self.shuffle else None,
            )
            folds = list(splitter.split(indices, y))
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state if self.shuffle else None,
            )
            folds = list(splitter.split(indices))
        fold_index = self.split or 0
        if fold_index >= len(folds):
            raise ValueError(
                f"split={fold_index} out of range for n_splits={self.n_splits}",
            )

        train_val_idx, val_idx = folds[fold_index]

        val_cap = self._to_count(self.val_size, len(indices))
        if val_cap is not None:
            val_cap = max(0, min(int(val_cap), len(indices)))
            if len(val_idx) > val_cap:
                stratify_val = y[val_idx] if y is not None else None
                val_idx, _ = train_test_split(
                    val_idx,
                    train_size=val_cap,
                    random_state=self.random_state,
                    stratify=stratify_val,
                )
            train_val_idx = np.setdiff1d(indices, val_idx)

        train_cap = self._to_count(self.train_size, len(indices))
        if train_cap is not None:
            train_cap = max(0, min(int(train_cap), len(train_val_idx)))
            if len(train_val_idx) > train_cap:
                stratify_train_pool = y[train_val_idx] if y is not None else None
                train_val_idx, _ = train_test_split(
                    train_val_idx,
                    train_size=train_cap,
                    random_state=self.random_state,
                    stratify=stratify_train_pool,
                )

        if (
            isinstance(self.test_size, int)
            and isinstance(self.train_size, int)
            and self.n_splits > 0
            and self.test_size > self.train_size // self.n_splits
        ):
            raise ValueError(
                "test_size must be <= train_size // n_splits for PytorchFoldSampler "
                f"(got test_size={self.test_size}, train_size={self.train_size}, n_splits={self.n_splits})",
            )

        stratify_sub = y[train_val_idx] if y is not None else None
        train_idx, test_idx = train_test_split(
            train_val_idx,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_sub,
        )
        return train_idx, test_idx, val_idx


@dataclass
class PytorchShuffleSampler(PytorchBaseSampler):
    """PytorchShuffleSampler runtime class.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    n_splits: int = 5
    split: int | None = 0
    test_size: int | float | None = 0.2
    val_size: int | float | None = None
    random_state: int | None = 42
    stratify: bool | str | None = True

    def __call__(self, config: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate shuffled train/test/validation splits.

        Args:
            config: Runtime data-like config.

        Returns:
            Train, test, and validation index arrays.

        Raises:
            ValueError: If validation size or split index is invalid.
        """
        self._validate_stratify(self.stratify)
        self._validate_fractional_sizes(None, self.test_size, self.val_size)
        if self.n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {self.n_splits}.")
        if self.val_size is None:
            raise ValueError("val_size must be set for PytorchShuffleSampler")
        dataset = getattr(config, "dataset_obj", None)
        if dataset is None:
            if (
                getattr(config, "_X", None) is None
                or getattr(config, "_y", None) is None
            ):
                raise ValueError("Data not loaded. Call load_dataset() first.")
            dataset = TensorDataset(config._X, config._y)
        indices = np.arange(len(dataset))
        labels = getattr(config, "_y", None)
        y = (
            labels.detach().cpu().numpy()
            if (self.stratify and labels is not None)
            else None
        )

        if y is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.val_size,
                random_state=self.random_state,
            )
            splits = list(splitter.split(indices, y))
        else:
            splitter = ShuffleSplit(
                n_splits=self.n_splits,
                test_size=self.val_size,
                random_state=self.random_state,
            )
            splits = list(splitter.split(indices))

        split_index = self.split if self.split is not None else 0
        if split_index >= len(splits):
            raise ValueError(
                f"split={split_index} out of range for n_splits={self.n_splits}",
            )

        train_test_idx, val_idx = splits[split_index]
        stratify_sub = y[train_test_idx] if y is not None else None

        train_idx, test_idx = train_test_split(
            train_test_idx,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_sub,
            shuffle=True,
        )
        return train_idx, test_idx, val_idx


PytorchKFoldSampler = PytorchFoldSampler
TorchBaseSampler = PytorchBaseSampler
TorchSplitSampler = PytorchSplitSampler
TorchKFoldSampler = PytorchFoldSampler
TorchShuffleSampler = PytorchShuffleSampler
