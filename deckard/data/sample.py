"""Pluggable sampler classes for :class:`~deckard.data.DataConfig`.

Each sampler is a callable dataclass that accepts a ``DataConfig`` instance and
returns ``(train_idx, test_idx, val_idx)`` as numpy arrays of integer indices.

Available samplers:

* :class:`BaseSampler` – abstract interface (raises ``NotImplementedError``)
* :class:`SplitSampler` – deterministic 3-way train/test/val split
* :class:`KFoldSampler` – cross-validation with disjoint validation folds
* :class:`ShuffleSampler` – repeated random (Monte-Carlo) splits

Hydra ConfigStore registration:

Calling :func:`register_sampler_configs` registers structured-config defaults for
all three concrete samplers under the ``sample`` Hydra config group. Defaults
are registered with names ``split``, ``kfold``, and ``shuffle``. A ``none``
entry (empty config / no sampler) is also registered for opt-out.

Example CLI usage::

    python -m deckard data=adult data@sample=kfold

"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from hydra.core.config_store import ConfigStore

if TYPE_CHECKING:
    from .base import DataConfig

# =========================================================
# Base interface
# =========================================================


@dataclass
class BaseSampler:
    """Callable sampler interface.

    All concrete samplers must implement :meth:`__call__` and return a
    ``(train_idx, test_idx, val_idx)`` triple of integer numpy arrays.
    """

    def __call__(
        self,
        config: "DataConfig",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


# =========================================================
# Split sampler (deterministic 3-way split)
# =========================================================


@dataclass
class SplitSampler(BaseSampler):
    """Standard 3-way stratified split: train / test / val.

    The dataset is first split into a *val* set (controlled by
    ``cfg.val_size``) and a remaining *train+test* pool. The pool is then
    split into *train* and *test* portions according to ``cfg.test_size``.

    Note: All parameters are read from the ``DataConfig`` passed to
    :meth:`__call__`. There are no constructor parameters.
    """

    def __call__(self, cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert cfg._X is not None, "Data must be loaded before sampling"
        indices = np.arange(len(cfg._X))
        stratify_col = cfg._get_stratify_col()

        if cfg.val_size is not None:
            # 3-way split: isolate validation set first
            train_test_idx, val_idx = train_test_split(
                indices,
                test_size=cfg.val_size,
                random_state=cfg.random_state,
                stratify=stratify_col if stratify_col is not None else None,
            )
            stratify_sub = (
                stratify_col.iloc[train_test_idx] if stratify_col is not None else None
            )
            train_idx, test_idx = train_test_split(
                train_test_idx,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
                stratify=stratify_sub,
            )
        else:
            # 2-way split: no validation set
            train_idx, test_idx = train_test_split(
                indices,
                train_size=cfg.train_size,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
                stratify=stratify_col if stratify_col is not None else None,
            )
            val_idx = np.array([], dtype=int)

        return train_idx, test_idx, val_idx


# =========================================================
# KFold sampler (disjoint validation folds)
# =========================================================


@dataclass
class KFoldSampler(BaseSampler):
    """Cross-validation sampler with disjoint validation folds.

     Behavior summary
     ----------------
     1. A fold is selected by ``cfg.split`` from ``n_splits`` CV folds.
     2. The selected fold is the initial validation set.
     3. ``cfg.val_size`` is treated as a *cap* on that validation set.
         - If the fold has more rows than ``val_size``, it is downsampled.
         - If the fold has fewer rows than ``val_size``, it is left unchanged.
     4. The non-validation pool is rebuilt as the complement of the (possibly
         capped) validation indices.
     5. ``cfg.train_size`` is treated as a *cap* on this non-validation pool.
         - If the pool is larger than ``train_size``, it is downsampled.
         - If smaller, it is left unchanged.
     6. The capped non-validation pool is split into train/test using
         ``cfg.test_size``.

     Guardrail
     ---------
     For integer sizing, this sampler enforces:

     ``test_size <= train_size // n_splits``

     and raises ``ValueError`` if violated.

     Notes
     -----
     - Under this sampler, ``train_size`` is the cap for the *train+test pool*
        before the final train/test split, not the final train row count.
     - The final train size is therefore typically:

        ``final_train = capped_train_pool - test_size``

     Example
     -------
     With ``n_samples=1200``, ``train_size=1000``, ``val_size=200``,
     ``test_size=200``, ``n_splits=5``:

     - validation cap: 200
     - capped train+test pool: 1000
     - final train/test/val per split: ``800 / 200 / 200``

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.
    shuffle : bool, default True
        Whether to shuffle the data before splitting into folds.
    """

    n_splits: int = 5
    shuffle: bool = True

    @staticmethod
    def _to_count(size, total: int):
        """Convert an int/float size spec to an absolute count."""
        if size is None:
            return None
        if isinstance(size, float):
            if size <= 0:
                return 0
            if size < 1:
                return int(np.floor(total * size))
            return int(size)
        return int(size)

    def __call__(self, cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert cfg._X is not None, "Data must be loaded before sampling"
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

        split = cfg.split if cfg.split is not None else 0
        if split >= len(splits):
            raise ValueError(
                f"split={split} out of range for n_splits={self.n_splits}",
            )

        train_val_idx, val_idx = splits[split]

        # Treat val_size as a cap on the selected fold validation set.
        val_cap = self._to_count(getattr(cfg, "val_size", None), len(indices))
        if val_cap is not None:
            val_cap = max(0, min(int(val_cap), len(indices)))
            if len(val_idx) > val_cap:
                stratify_val = (
                    stratify_col.iloc[val_idx] if stratify_col is not None else None
                )
                val_idx, _ = train_test_split(
                    val_idx,
                    train_size=val_cap,
                    random_state=cfg.random_state,
                    stratify=stratify_val,
                )

            # Rebuild the non-validation pool after capping val size.
            train_val_idx = np.setdiff1d(indices, val_idx)

        # Treat train_size as a cap on the non-validation pool.
        train_cap = self._to_count(getattr(cfg, "train_size", None), len(indices))
        if train_cap is not None:
            train_cap = max(0, min(int(train_cap), len(train_val_idx)))
            if len(train_val_idx) > train_cap:
                stratify_train_pool = (
                    stratify_col.iloc[train_val_idx]
                    if stratify_col is not None
                    else None
                )
                train_val_idx, _ = train_test_split(
                    train_val_idx,
                    train_size=train_cap,
                    random_state=cfg.random_state,
                    stratify=stratify_train_pool,
                )

        # User-requested guardrail for explicit integer sizing.
        if (
            isinstance(getattr(cfg, "test_size", None), int)
            and isinstance(getattr(cfg, "train_size", None), int)
            and self.n_splits > 0
        ):
            if cfg.test_size > cfg.train_size // self.n_splits:
                raise ValueError(
                    "test_size must be <= train_size // n_splits for KFoldSampler "
                    f"(got test_size={cfg.test_size}, train_size={cfg.train_size}, n_splits={self.n_splits})",
                )

        # Stratification for the inner train/test split
        stratify_sub = (
            stratify_col.iloc[train_val_idx] if stratify_col is not None else None
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

    def __call__(self, cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cfg.val_size is None:
            raise ValueError("val_size must be set for ShuffleSampler")

        assert cfg._X is not None, "Data must be loaded before sampling"
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

        split = cfg.split if cfg.split is not None else 0
        if split >= len(splits):
            raise ValueError(
                f"split={split} out of range for n_splits={self.n_splits}",
            )

        train_test_idx, val_idx = splits[split]

        # Stratification for the inner train/test split
        stratify_sub = (
            stratify_col.iloc[train_test_idx] if stratify_col is not None else None
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
