from __future__ import annotations

# Standard library
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

# Third-party
import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from torch import Tensor
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    TensorDataset,
    random_split,
)

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
    "TorchDataLoaderMixin",
    "TorchDataLoaderSamplingMixin",
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

        def _cfg_value(name: str, default: Any) -> Any:
            getter = getattr(config, "_get_sampler_option", None)
            if callable(getter):
                return getter(name, default)
            if name in sampler_params:
                return sampler_params[name]
            return default

        def _sampler_kwargs_for_alias(alias: str) -> dict[str, Any]:
            if alias == "split":
                return {
                    "train_size": _cfg_value("train_size", None),
                    "test_size": _cfg_value("test_size", None),
                    "val_size": _cfg_value("val_size", None),
                    "random_state": _cfg_value("random_state", 42),
                    "stratify": _cfg_value("stratify", True),
                }
            if alias in {"fold", "kfold"}:
                return {
                    "n_splits": getattr(config, "n_splits", 5),
                    "split": getattr(config, "split", 0),
                    "train_size": _cfg_value("train_size", None),
                    "test_size": _cfg_value("test_size", None),
                    "val_size": _cfg_value("val_size", None),
                    "random_state": _cfg_value("random_state", 42),
                    "stratify": _cfg_value("stratify", True),
                }
            if alias == "shuffle":
                return {
                    "n_splits": getattr(config, "n_splits", 5),
                    "split": getattr(config, "split", 0),
                    "test_size": _cfg_value("test_size", None),
                    "val_size": _cfg_value("val_size", None),
                    "random_state": _cfg_value("random_state", 42),
                    "stratify": _cfg_value("stratify", True),
                }
            return {}

        spec = getattr(config, "sampler", None)
        if spec is None:
            return None

        if isinstance(spec, str):
            key = spec.strip().lower()
            if key not in sampler_aliases:
                raise ValueError(
                    f"Unknown sampler '{spec}'. Must be one of {list(sampler_aliases)}.",
                )
            alias_kwargs = _sampler_kwargs_for_alias(key)
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
                alias_kwargs = _sampler_kwargs_for_alias(key)
                alias_kwargs.update(sampler_params)
                alias_kwargs.update({str(k): v for k, v in spec.items()})
                return sampler_aliases[key](**alias_kwargs)

            loaded = load_class(class_path, **{str(k): v for k, v in spec.items()})
            if isinstance(loaded, type):
                return loaded()
            return loaded

        if callable(spec) and not isinstance(spec, type):
            return spec

        if isinstance(spec, type):
            return spec()

        raise ValueError(f"Unsupported sampler specification: {type(spec)}")

    @classmethod
    def compose(cls, config: Any) -> Any:
        """Compose and cache runtime sampler callable.

        Args:
            config: Runtime data-like config.

        Returns:
            Callable sampler object.

        Raises:
            TypeError: If resolved sampler is not callable.
        """
        sampler_obj = getattr(config, "_sampler_obj", None)
        if sampler_obj is None:
            sampler_obj = cls.resolve(config)
            setattr(config, "_sampler_obj", sampler_obj)
        if sampler_obj is None:
            sampler_obj = PytorchSplitSampler()
            setattr(config, "_sampler_obj", sampler_obj)
        params = dict(getattr(config, "sampler_params", {}) or {})
        for field_name, value in params.items():
            if hasattr(sampler_obj, field_name):
                setattr(sampler_obj, field_name, value)
        if not callable(sampler_obj):
            raise TypeError(
                f"Composed sampler must be callable, got {type(sampler_obj)}",
            )
        return sampler_obj

    @classmethod
    def execute(cls, config: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve/compose and execute sampler for runtime config.

        Args:
            config: Runtime data-like config.

        Returns:
            Train, test, and validation index arrays.
        """
        sampler_obj = cls.compose(config)
        return sampler_obj(config)

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
                        f"{name} float value must be in [0, 1], got {value}."
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
        if size is None:
            return None
        if isinstance(size, float):
            if size <= 0:
                return 0
            if size < 1:
                return int(np.floor(total * size))
            return int(size)
        return int(size)

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


class TorchDataLoaderMixin:
    """Adapter that overloads `.sample()` for PyTorch DataLoaders.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    test_size: float
    val_size: float
    random_seed: int
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool

    def sample(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build train/test DataLoaders.


        Returns:
            (train_loader, test_loader)
        """
        dataset = getattr(self, "dataset", None)
        if dataset is None:

            if not hasattr(self, "X") or not hasattr(self, "y"):
                self.load_data()

            if not isinstance(self.X, torch.Tensor):
                self.X = torch.as_tensor(self.X)

            if not isinstance(self.y, torch.Tensor):
                self.y = torch.as_tensor(self.y)

            dataset = TensorDataset(self.X, self.y)

        n_total = len(dataset)
        n_test = int(n_total * self.test_size)
        n_val = int(n_total * self.val_size) if self.val_size else 0
        n_train = n_total - n_test - n_val
        generator = torch.Generator().manual_seed(self.random_seed)

        train_ds, test_ds, val_ds = random_split(
            dataset,
            [n_train, n_test, n_val],
            generator=generator,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
        return self.train_loader, self.test_loader, self.val_loader


class TorchDataLoaderSamplingMixin:
    """Sampling adapter for torch Dataset objects.

    Required attrs:
        dataset: Dataset
        test_size: float
        train_size: float
        val_size: float
        random_state: int
        sample: Literal["split", "fold", "shuffle"]
        stratify: bool

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    dataset: Dataset
    test_size: float
    train_size: float
    val_size: float
    random_state: int
    sample: Literal["split", "fold", "shuffle"]
    stratify: bool

    def _get_targets(self) -> list:
        """Extract labels for stratified sampling."""
        ds = self.dataset

        if hasattr(ds, "targets"):
            return list(ds.targets)

        if hasattr(ds, "labels"):
            return list(ds.labels)

        raise AttributeError(
            "Stratified sampling requires dataset.targets or dataset.labels.",
        )

    def _make_loader(
        self,
        subset: Dataset,
        *,
        shuffle: bool,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool = False,
    ) -> DataLoader:
        """Construct a DataLoader."""
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def sample(
        self,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        n_splits: int = 5,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Sample the dataset according to `self.sample`.

        Args:
            batch_size: Batch size for created dataloaders.
            num_workers: Number of dataloader workers.
            pin_memory: Whether to pin memory in dataloaders.
            drop_last: Whether to drop incomplete trailing batch in train loader.
            n_splits: Number of fold candidates for fold/shuffle modes.

        Returns:
            Train, validation, and test dataloaders.

        Raises:
            TypeError: If configured dataset is not a torch Dataset.
        """
        ds = self.dataset

        if not isinstance(ds, Dataset):
            raise TypeError(
                "dataset must be torch.utils.data.Dataset",
            )
        labels = None
        if getattr(self, "stratify", False):
            labels = torch.as_tensor(self._get_targets())

        sampler_cfg = SimpleNamespace(
            sampler=getattr(self, "sample", "split"),
            train_size=getattr(self, "train_size", None),
            test_size=getattr(self, "test_size", None),
            val_size=getattr(self, "val_size", None),
            random_state=getattr(self, "random_state", 42),
            stratify=getattr(self, "stratify", True),
            n_splits=n_splits,
            split=0,
        )

        sampler_cfg.dataset_obj = ds
        sampler_cfg._y = labels
        train_idx, test_idx, val_idx = PytorchBaseSampler.execute(sampler_cfg)

        self.train_loader = self._make_loader(
            Subset(ds, train_idx.tolist()),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        self.test_loader = self._make_loader(
            Subset(ds, test_idx.tolist()),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if len(val_idx) > 0:
            self.val_loader = self._make_loader(
                Subset(ds, val_idx.tolist()),
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            self.val_loader = self._make_loader(
                Subset(ds, []),
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        return self.train_loader, self.val_loader, self.test_loader

    def _validate_sizes(self) -> None:
        """Validate configured split sizes."""
        total = self.train_size + self.val_size + self.test_size

        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "train_size + val_size + test_size must equal 1.0",
            )

    def _sample_split(
        self,
        ds: Dataset,
        indices: list[int],
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test DataLoaders."""
        y = self._get_targets() if self.stratify else None

        # Step 1: isolate test set
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Step 2: split remaining into train/val
        y_trainval = [y[i] for i in trainval_idx] if y is not None else None

        val_fraction = self.val_size / (self.train_size + self.val_size)

        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_fraction,
            random_state=self.random_state,
            stratify=y_trainval,
        )

        self.train_loader = self._make_loader(
            Subset(ds, train_idx),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        self.val_loader = self._make_loader(
            Subset(ds, val_idx),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.test_loader = self._make_loader(
            Subset(ds, test_idx),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        )

    def _sample_fold(
        self,
        ds: Dataset,
        indices: list[int],
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        n_splits: int,
    ) -> list[tuple[DataLoader, DataLoader]]:
        """Create K-fold train/validation loaders."""
        y = self._get_targets() if self.stratify else None

        splitter = (
            StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            if self.stratify
            else KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
        )

        split_iter = (
            splitter.split(indices, y) if self.stratify else splitter.split(indices)
        )

        folds: list[tuple[DataLoader, DataLoader]] = []

        for train_idx, val_idx in split_iter:
            train_loader = self._make_loader(
                Subset(ds, train_idx),
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )

            val_loader = self._make_loader(
                Subset(ds, val_idx),
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

            folds.append((train_loader, val_loader))

        self.folds = folds
        return folds

    def _sample_shuffle(
        self,
        ds: Dataset,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
    ) -> DataLoader:
        """Create one shuffled DataLoader over the full dataset."""
        self.loader = self._make_loader(
            ds,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return self.loader
