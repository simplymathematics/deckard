"""Pluggable sampler classes for :class:`~deckard.data.DataConfig`.

Each sampler is a callable dataclass that accepts a ``DataConfig`` instance and
returns ``(train_idx, test_idx, val_idx)`` as numpy arrays of integer indices.

Samplers
--------
- :class:`BaseSampler`     – abstract interface (raises ``NotImplementedError``)
- :class:`SplitSampler`    – deterministic 3-way train / test / val split
- :class:`KFoldSampler`    – cross-validation with disjoint validation folds
- :class:`ShuffleSampler`  – repeated random (Monte-Carlo) splits

Hydra ConfigStore
-----------------
Calling :func:`register_sampler_configs` registers structured-config defaults for
all three concrete samplers under the ``sample`` Hydra config group.  The defaults
are registered with names ``split``, ``kfold``, and ``shuffle``.  A ``none``
entry (empty config / no sampler) is also registered so you can opt out explicitly.

Example CLI usage::

    python -m deckard data=adult sample=kfold

"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)


# =========================================================
# Base interface
# =========================================================


@dataclass
class BaseSampler:
    """Callable sampler interface.

    All concrete samplers must implement :meth:`__call__` and return a
    ``(train_idx, test_idx, val_idx)`` triple of integer numpy arrays.
    """

    def __call__(self, config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


# =========================================================
# Split sampler (deterministic 3-way split)
# =========================================================


@dataclass
class SplitSampler(BaseSampler):
    """Standard 3-way stratified split: train / test / val.

    The dataset is first split into a *val* set (controlled by
    ``cfg.val_size``) and a remaining *train+test* pool.  The pool is then
    split into *train* and *test* portions according to ``cfg.test_size``.

    Parameters
    ----------
    (none – all parameters are read from the ``DataConfig`` passed to
    :meth:`__call__`)
    """

    def __call__(self, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cfg.val_size is None:
            raise ValueError("val_size must be set for SplitSampler")

        indices = np.arange(len(cfg._X))
        stratify_col = cfg._get_stratify_col()

        # First split: isolate validation set
        train_test_idx, val_idx = train_test_split(
            indices,
            test_size=cfg.val_size,
            random_state=cfg.random_state,
            stratify=stratify_col if stratify_col is not None else None,
        )

        # Adjust stratification for the inner split
        stratify_sub = (
            stratify_col.iloc[train_test_idx]
            if stratify_col is not None
            else None
        )

        # Second split: train vs test
        train_idx, test_idx = train_test_split(
            train_test_idx,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify_sub,
        )

        return train_idx, test_idx, val_idx


# =========================================================
# KFold sampler (disjoint validation folds)
# =========================================================


@dataclass
class KFoldSampler(BaseSampler):
    """Cross-validation sampler with disjoint validation folds.

    The val set is the fold selected by ``cfg.fold``; the remaining data is
    split into train and test portions according to ``cfg.test_size``.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    shuffle : bool, default True
        Whether to shuffle the data before splitting into folds.
    """

    n_splits: int = 5
    shuffle: bool = True

    def __call__(self, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.arange(len(cfg._X))
        stratify_col = cfg._get_stratify_col()

        # Choose stratified or plain splitter
        if stratify_col is not None:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=cfg.random_state if self.shuffle else None,
            )
            splits = list(splitter.split(indices, stratify_col))
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=cfg.random_state if self.shuffle else None,
            )
            splits = list(splitter.split(indices))

        fold = cfg.fold if cfg.fold is not None else 0
        if fold >= len(splits):
            raise ValueError(
                f"fold={fold} out of range for n_splits={self.n_splits}"
            )

        train_val_idx, val_idx = splits[fold]

        # Stratification for the inner train/test split
        stratify_sub = (
            stratify_col.iloc[train_val_idx]
            if stratify_col is not None
            else None
        )

        train_idx, test_idx = train_test_split(
            train_val_idx,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify_sub,
        )

        return train_idx, test_idx, val_idx


# =========================================================
# Shuffle sampler (Monte Carlo CV)
# =========================================================


@dataclass
class ShuffleSampler(BaseSampler):
    """Repeated random-split (Monte-Carlo) sampler.

    Each fold is an independent random split; the val set is *not*
    guaranteed to be disjoint across folds.

    Parameters
    ----------
    n_splits : int, default 5
        Number of re-shuffled splits to generate.
    """

    n_splits: int = 5

    def __call__(self, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cfg.val_size is None:
            raise ValueError("val_size must be set for ShuffleSampler")

        indices = np.arange(len(cfg._X))
        stratify_col = cfg._get_stratify_col()

        # Choose stratified or plain splitter
        if stratify_col is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=self.n_splits,
                test_size=cfg.val_size,
                random_state=cfg.random_state,
            )
            splits = list(splitter.split(indices, stratify_col))
        else:
            splitter = ShuffleSplit(
                n_splits=self.n_splits,
                test_size=cfg.val_size,
                random_state=cfg.random_state,
            )
            splits = list(splitter.split(indices))

        fold = cfg.fold if cfg.fold is not None else 0
        if fold >= len(splits):
            raise ValueError(
                f"fold={fold} out of range for n_splits={self.n_splits}"
            )

        train_test_idx, val_idx = splits[fold]

        # Stratification for the inner train/test split
        stratify_sub = (
            stratify_col.iloc[train_test_idx]
            if stratify_col is not None
            else None
        )

        train_idx, test_idx = train_test_split(
            train_test_idx,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify_sub,
        )

        return train_idx, test_idx, val_idx


# =========================================================
# Hydra structured config dataclasses
# =========================================================


@dataclass
class SplitSamplerConf:
    """Hydra structured config for :class:`SplitSampler`.

    Register with the ``sample`` config group via
    :func:`register_sampler_configs`.
    """

    name: str = "deckard.data.sample.SplitSampler"


@dataclass
class KFoldSamplerConf:
    """Hydra structured config for :class:`KFoldSampler`.

    Register with the ``sample`` config group via
    :func:`register_sampler_configs`.
    """

    name: str = "deckard.data.sample.KFoldSampler"
    n_splits: int = 5
    shuffle: bool = True


@dataclass
class ShuffleSamplerConf:
    """Hydra structured config for :class:`ShuffleSampler`.

    Register with the ``sample`` config group via
    :func:`register_sampler_configs`.
    """

    name: str = "deckard.data.sample.ShuffleSampler"
    n_splits: int = 5


def register_sampler_configs() -> None:
    """Register sampler structured configs with the Hydra ConfigStore.

    Call this function once at application startup (e.g. in your ``@hydra.main``
    script) to make the ``sample`` config group available.

    After calling this function, the following CLI overrides are available::

        sample=split
        sample=kfold
        sample=shuffle
        sample=none    # disables the sampler (legacy 2-way split)

    When a sampler is selected, the config is placed under ``data.sample``
    via the ``@data.sample`` package override.
    """
    try:
        from hydra.core.config_store import ConfigStore
    except ImportError:  # pragma: no cover
        return

    cs = ConfigStore.instance()
    cs.store(
        group="sample",
        name="split",
        node=SplitSamplerConf,
        package="data.sample",
    )
    cs.store(
        group="sample",
        name="kfold",
        node=KFoldSamplerConf,
        package="data.sample",
    )
    cs.store(
        group="sample",
        name="shuffle",
        node=ShuffleSamplerConf,
        package="data.sample",
    )
    # 'none' leaves data.sample as None (no sampler, legacy behavior)
    cs.store(
        group="sample",
        name="none",
        node={},
        package="data.sample",
    )
