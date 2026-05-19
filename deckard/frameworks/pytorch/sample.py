from __future__ import annotations

# Standard library
import logging
from dataclasses import dataclass
from typing import Literal

# Third-party
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

# Local / project

# Logger
logger = logging.getLogger(__name__)


from torch.utils.data import (
    TensorDataset,
    random_split,
)

MatrixLike = Tensor
ArrayLike = Tensor




class TorchDataLoaderMixin:
    """Adapter that overloads `.sample()` for PyTorch DataLoaders."""

    def sample(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build train/test DataLoaders.


        Returns:
            (train_loader, test_loader)
        """
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
        if self.val_size:
            n_test = n_total - self.val_size
        else:
            n_val = 0
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
            pin_memory=self.in_memory,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
    ):
        """Sample the dataset according to `self.sample`.

        Modes:
            split   -> (train_loader, val_loader, test_loader)
            fold    -> list[(train_loader, val_loader)]
            shuffle -> single shuffled DataLoader
        """
        ds = self.dataset

        if not isinstance(ds, Dataset):
            raise TypeError(
                "dataset must be torch.utils.data.Dataset",
            )

        self._validate_sizes()

        indices = list(range(len(ds)))

        if self.sample == "split":
            return self._sample_split(
                ds,
                indices,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )

        if self.sample == "fold":
            return self._sample_fold(
                ds,
                indices,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                n_splits=n_splits,
            )

        if self.sample == "shuffle":
            return self._sample_shuffle(
                ds,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )

        raise ValueError(
            "sample must be 'split', 'fold', or 'shuffle'",
        )

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
        return self.loader @ dataclass(eq=False, kw_only=True)
