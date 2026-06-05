"""Pluggable sampler classes for :class:`~deckard.data.DataConfig`.

Each sampler is a callable dataclass that accepts a ``DataConfig`` instance and
returns ``(train_idx, test_idx, val_idx)`` as numpy arrays of integer indices.

Available samplers:

* :class:`BaseSampler` – interface plus centralized sampler
    resolution/composition/execution helpers
* :class:`SplitSampler` – deterministic 3-way train/test/val split
* :class:`KFoldSampler` – cross-validation with disjoint validation folds
* :class:`ShuffleSampler` – repeated random (Monte-Carlo) splits

Hydra ConfigStore registration:

Calling :func:`register_sampler_configs` registers structured-config defaults for
all three concrete samplers under the ``sample`` Hydra config group. Defaults
are registered with names ``split``, ``kfold``, and ``shuffle``. A ``none``
entry (empty config / no sampler) is also registered for opt-out.

Example CLI usage:

```text
python -m deckard data=adult data@sample=kfold
```

"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np
from hydra.core.config_store import ConfigStore
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)

from ..utils import resolve_component_spec

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

    This class also owns runtime sampling orchestration for ``DataConfig`` via:
    - :meth:`resolve` to normalize sampler configuration into a callable
    - :meth:`compose` to resolve/cache/fallback to a default sampler
    - :meth:`execute` to run the composed sampler

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    def __call__(
        self,
        config: "DataConfig",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run sampling strategy against runtime data config.

        Args:
            config: Runtime data config.

        Returns:
            Train, test, and validation index arrays.

        Raises:
            NotImplementedError: Always raised by base sampler interface.
        """
        raise NotImplementedError

    @staticmethod
    def _get_stratify_col(
        config: "DataConfig",
        stratify: bool | str | None = None,
    ):
        """Return stratification labels for sampler execution.

        Stratify behavior is sampler-owned and optionally sourced from sampler
        configuration when ``stratify`` is not explicitly provided.
        """
        if stratify is None:
            sampler_cfg = getattr(config, "sampler", None)
            if isinstance(sampler_cfg, dict):
                stratify = sampler_cfg.get("stratify", None)
            elif hasattr(sampler_cfg, "stratify"):
                stratify = getattr(sampler_cfg, "stratify")

        if stratify is None or stratify is False:
            return None
        if stratify is True:
            if getattr(config, "classifier", None) is False:
                return None
            return getattr(config, "_y", None)
        if isinstance(stratify, str):
            X = getattr(config, "_X", None)
            if X is None:
                raise ValueError("_X not set. Load the data before sampling.")
            columns = getattr(X, "columns", None)
            if columns is not None and stratify in columns:
                return X[stratify]
            raise ValueError(
                f"Stratify column '{stratify}' not found in data columns",
            )
        raise ValueError("stratify must be None, True, False, or a column name")

    @staticmethod
    def _size_to_count(size, total: int):
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

    @staticmethod
    def _resolve_terminal_sampler_spec(spec: Any) -> Any:
        """Resolve callable/type sampler specs or raise for unsupported specs."""
        if callable(spec) and not isinstance(spec, type):
            return spec
        if isinstance(spec, type):
            return spec()
        raise ValueError(f"Unsupported sampler specification: {type(spec)}")

    @staticmethod
    def _sampler_spec_or_none(config: "DataConfig") -> Any:
        spec = getattr(config, "sampler", None)
        if spec is None:
            return None
        if isinstance(spec, dict) and not spec:
            return None
        return spec

    @staticmethod
    def _resolve_component(
        *,
        spec: Any,
        aliases: dict[str, Any],
        alias_kwargs_getter,
    ) -> Any:
        return resolve_component_spec(
            spec,
            field_name="sampler",
            aliases=aliases,
            alias_kwargs_getter=alias_kwargs_getter,
        )

    @classmethod
    def resolve(cls, config: "DataConfig") -> Any:
        """Resolve ``config.sampler`` into a callable sampler object or ``None``.

        Args:
            config: Runtime data config with sampler declaration.

        Returns:
            Resolved sampler object or ``None``.

        Raises:
            ValueError: If sampler declaration is invalid or unsupported.
        """
        sampler_aliases = {
            "split": SplitSampler,
            "fold": KFoldSampler,
            "kfold": KFoldSampler,
            "shuffle": ShuffleSampler,
        }

        def _alias_kwargs(source: Any, alias: str) -> dict[str, Any]:
            if isinstance(source, dict):
                return {str(key): value for key, value in source.items()}
            return {}

        spec = cls._sampler_spec_or_none(config)
        if spec is None:
            return None

        if isinstance(spec, dict):
            payload = dict(spec)
            class_path = payload.pop("name", payload.pop("_target_", None))
            if class_path is None:
                return SplitSampler(
                    **{str(key): value for key, value in payload.items()},
                )

        return cls._resolve_component(
            spec=spec,
            aliases=sampler_aliases,
            alias_kwargs_getter=_alias_kwargs,
        )

    @classmethod
    def _default_sampler(cls) -> Any:
        """Return the default sampler instance when no sampler is configured."""
        return SplitSampler()

    @classmethod
    def _configure_composed_sampler(
        cls,
        sampler_obj: Any,
        config: "DataConfig",
    ) -> Any:
        """Apply framework-specific sampler configuration after composition."""
        _ = config
        return sampler_obj

    @classmethod
    def compose(cls, config: "DataConfig") -> Any:
        """Compose and cache the runtime sampler callable for ``config``.

        Args:
            config: Runtime data config.

        Returns:
            Callable sampler object.

        Raises:
            TypeError: If composed sampler is not callable.
        """
        sampler_obj = getattr(config, "_sampler_obj", None)
        if sampler_obj is None:
            sampler_obj = cls.resolve(config)
            setattr(config, "_sampler_obj", sampler_obj)
        if sampler_obj is None:
            sampler_obj = cls._default_sampler()
            setattr(config, "_sampler_obj", sampler_obj)
        sampler_obj = cls._configure_composed_sampler(sampler_obj, config)
        setattr(config, "_sampler_obj", sampler_obj)
        if not callable(sampler_obj):
            raise TypeError(
                f"Composed sampler must be callable, got {type(sampler_obj)}",
            )
        return sampler_obj

    @classmethod
    def execute(
        cls,
        config: "DataConfig",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve/compose and run the configured sampler against ``config``.

        Args:
            config: Runtime data config.

        Returns:
            Train, test, and validation index arrays.
        """
        sampler_obj = cls.compose(config)
        return sampler_obj(config)


# =========================================================
# Split sampler (deterministic 3-way split)
# =========================================================


@dataclass
class SplitSampler(BaseSampler):
    """Standard 3-way stratified split: train / test / val.

    The dataset is first split into a *val* set (controlled by
    ``cfg.val_size``) and a remaining *train+test* pool. The pool is then
    split into *train* and *test* portions according to ``cfg.test_size``.

    Parameters are owned by this sampler dataclass and configured directly on
    the sampler instance (or its Hydra dict/spec).

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    train_size: int | float | None = None
    test_size: int | float | None = 0.2
    val_size: int | float | None = None
    random_state: int | None = 42
    stratify: bool | str | None = True

    def __call__(self, cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate train/test/validation splits.

        Args:
            cfg: Runtime data config.

        Returns:
            Train, test, and validation index arrays.
        """
        assert cfg._X is not None, "Data must be loaded before sampling"
        indices = np.arange(len(cfg._X))
        train_size = self.train_size
        test_size = self.test_size
        if isinstance(self.val_size, (int, float)) and self.val_size > 0:
            val_size = self.val_size
        else:
            val_size = None
        random_state = self.random_state
        stratify_col = self._get_stratify_col(cfg, self.stratify)

        if val_size is not None:
            # 3-way split: isolate validation set first
            train_test_idx, val_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=random_state,
                stratify=stratify_col if stratify_col is not None else None,
            )
            stratify_sub = (
                stratify_col.iloc[train_test_idx] if stratify_col is not None else None
            )
            train_idx, test_idx = train_test_split(
                train_test_idx,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_sub,
            )
        else:
            # 2-way split: no validation set
            train_idx, test_idx = train_test_split(
                indices,
                train_size=train_size,
                test_size=test_size,
                random_state=random_state,
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

    Behavior summary:
    - Select fold ``cfg.split`` from ``n_splits`` CV folds as validation.
    - Treat ``cfg.val_size`` as a cap on validation rows.
    - Rebuild the non-validation pool after capping validation rows.
    - Treat ``cfg.train_size`` as a cap on the non-validation pool.
    - Split capped non-validation rows into train/test by ``cfg.test_size``.

    Guardrail:
    - For integer sizing, enforce ``test_size <= train_size // n_splits``.

    Notes:
    - ``train_size`` caps the train+test pool before the final split.
    - Final train size is typically ``capped_train_pool - test_size``.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    n_splits: int = 5
    shuffle: bool = True
    split: int | None = 0
    train_size: int | float | None = None
    test_size: int | float | None = 0.2
    val_size: int | float | None = None
    random_state: int | None = 42
    stratify: bool | str | None = True

    @staticmethod
    def _to_count(size, total: int):
        """Convert an int/float size spec to an absolute count."""
        return BaseSampler._size_to_count(size, total)

    def __call__(self, cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate fold-based train/test/validation splits.

        Args:
            cfg: Runtime data config.

        Returns:
            Train, test, and validation index arrays.

        Raises:
            ValueError: If selected split is out of range or sizing is invalid.
        """
        assert cfg._X is not None, "Data must be loaded before sampling"
        indices = np.arange(len(cfg._X))
        split = self.split if self.split is not None else 0
        train_size = self.train_size
        test_size = self.test_size
        val_size = self.val_size
        random_state = self.random_state

        stratify_col = self._get_stratify_col(cfg, self.stratify)

        # Choose stratified or plain splitter
        if stratify_col is not None:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=random_state if self.shuffle else None,
            )
            splits = list(splitter.split(indices, stratify_col))
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=random_state if self.shuffle else None,
            )
            splits = list(splitter.split(indices))

        split = split if split is not None else 0
        if split >= len(splits):
            raise ValueError(
                f"split={split} out of range for n_splits={self.n_splits}",
            )

        train_val_idx, val_idx = splits[split]

        # Treat val_size as a cap on the selected fold validation set.
        val_cap = self._to_count(val_size, len(indices))
        if val_cap is not None:
            val_cap = max(0, min(int(val_cap), len(indices)))
            if len(val_idx) > val_cap:
                stratify_val = (
                    stratify_col.iloc[val_idx] if stratify_col is not None else None
                )
                val_idx, _ = train_test_split(
                    val_idx,
                    train_size=val_cap,
                    random_state=random_state,
                    stratify=stratify_val,
                )

            # Rebuild the non-validation pool after capping val size.
            train_val_idx = np.setdiff1d(indices, val_idx)

        # Treat train_size as a cap on the non-validation pool.
        train_cap = self._to_count(train_size, len(indices))
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
                    random_state=random_state,
                    stratify=stratify_train_pool,
                )

        # User-requested guardrail for explicit integer sizing.
        if (
            isinstance(test_size, int)
            and isinstance(train_size, int)
            and self.n_splits > 0
        ):
            if test_size > train_size // self.n_splits:
                raise ValueError(
                    "test_size must be <= train_size // n_splits for KFoldSampler "
                    f"(got test_size={test_size}, train_size={train_size}, n_splits={self.n_splits})",
                )

        # Stratification for the inner train/test split
        stratify_sub = (
            stratify_col.iloc[train_val_idx] if stratify_col is not None else None
        )

        train_idx, test_idx = train_test_split(
            train_val_idx,
            test_size=test_size,
            random_state=random_state,
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

    The ``n_splits`` field controls how many re-shuffled splits are generated.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    n_splits: int = 5
    split: int | None = 0
    test_size: int | float | None = 0.2
    val_size: int | float | None = None
    random_state: int | None = 42
    stratify: bool | str | None = True

    def __call__(self, cfg: "DataConfig") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate repeated random train/test/validation splits.

        Args:
            cfg: Runtime data config.

        Returns:
            Train, test, and validation index arrays.

        Raises:
            ValueError: If ``val_size`` is missing or selected split is out of range.
        """
        split = self.split if self.split is not None else 0
        test_size = self.test_size
        val_size = self.val_size
        random_state = self.random_state

        if val_size is None:
            raise ValueError("val_size must be set for ShuffleSampler")

        assert cfg._X is not None, "Data must be loaded before sampling"
        indices = np.arange(len(cfg._X))
        stratify_col = self._get_stratify_col(cfg, self.stratify)

        # Choose stratified or plain splitter
        if stratify_col is not None:
            splitter = StratifiedShuffleSplit(
                n_splits=self.n_splits,
                test_size=val_size,
                random_state=random_state,
            )
            splits = list(splitter.split(indices, stratify_col))
        else:
            splitter = ShuffleSplit(
                n_splits=self.n_splits,
                test_size=val_size,
                random_state=random_state,
            )
            splits = list(splitter.split(indices))

        split = split if split is not None else 0
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
            test_size=test_size,
            random_state=random_state,
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

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    name: str = "deckard.data.sample.SplitSampler"
    train_size: int | float | None = None
    test_size: int | float | None = None
    val_size: int | float | None = None
    random_state: int | None = None
    stratify: bool | str | None = None


@dataclass
class KFoldSamplerConf:
    """Hydra structured config for :class:`KFoldSampler`.

    Register with the ``sample`` config group via
    :func:`register_sampler_configs`.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    name: str = "deckard.data.sample.KFoldSampler"
    n_splits: int = 5
    shuffle: bool = True
    split: int | None = None
    train_size: int | float | None = None
    test_size: int | float | None = None
    val_size: int | float | None = None
    random_state: int | None = None
    stratify: bool | str | None = None


@dataclass
class ShuffleSamplerConf:
    """Hydra structured config for :class:`ShuffleSampler`.

    Register with the ``sample`` config group via
    :func:`register_sampler_configs`.

    Attributes:
        Runtime attributes are inherited or configured via class fields documented in this module.
    """

    name: str = "deckard.data.sample.ShuffleSampler"
    n_splits: int = 5
    split: int | None = None
    test_size: int | float | None = None
    val_size: int | float | None = None
    random_state: int | None = None
    stratify: bool | str | None = None


def register_sampler_configs() -> None:
    """Register sampler structured configs with the Hydra ConfigStore.

    Call this function once at application startup (e.g. in your ``@hydra.main``
    script) to make the ``sampler`` config group available.

    Example:

    ```text
    sampler=split
    sampler=kfold
    sampler=shuffle
    sampler=none    # disables the sampler (legacy 2-way split)
    ```

    When a sampler is selected, the config is placed under ``data.sampler``
    via the ``@data.sampler`` package override.
    """

    cs = ConfigStore.instance()
    cs.store(
        group="sampler",
        name="split",
        node=SplitSamplerConf,
        package="data.sampler",
    )
    cs.store(
        group="sampler",
        name="kfold",
        node=KFoldSamplerConf,
        package="data.sampler",
    )
    cs.store(
        group="sampler",
        name="shuffle",
        node=ShuffleSamplerConf,
        package="data.sampler",
    )
    # 'none' leaves data.sampler as None (no sampler, legacy behavior)
    cs.store(
        group="sampler",
        name="none",
        node={},
        package="data.sampler",
    )
