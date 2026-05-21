# Imports
import os
import pandas as pd
import time
import logging
from pathlib import Path

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Tuple, Union, Optional
from omegaconf import DictConfig, ListConfig

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer

from scipy.sparse import csr_matrix

# deckard
from ..utils import (
    ConfigBase,
    data_supported_filetypes,
    load_class,
    coerce_to_list,
    merge_list_of_dicts,
    normalize_plugin_specs,
    instantiate_plugin_spec,
)
from ..frameworks.types import ArrayLike, MatrixLike
from ._mixins import DataPipelineMixin
from .stages import normalize_data_score_stage, stage_hook_token

# Setup logger
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .sample import BaseSampler


AUTO_SCORER = "auto"
DECKARD_TEST_MAX_SAMPLES_ENV = "DECKARD_TEST_MAX_SAMPLES"


def _coerce_scorer_config(*args, **kwargs):
    """Lazy import scorer coercion to avoid data<->score import cycles at module import time."""
    from ..score.base import coerce_scorer_config as _coerce

    return _coerce(*args, **kwargs)


def _is_data_scorer_instance(scorer: Any) -> bool:
    """Return True when *scorer* is explicitly a data-profile scorer."""
    from ..score.base import _DataScorerMarker

    if isinstance(scorer, _DataScorerMarker):
        return True
    if callable(scorer):
        return True
    return str(getattr(scorer, "scoring_type", "")).strip().lower() == "data"


@dataclass(eq=False, kw_only=True)
class DataHookPlugin:
    """Generic data plugin that delegates one hook to one runtime method.

    Initialization fields
    ---------------------
    hook_name : str
        Hook method name exposed to runtime (e.g., ``before_sample``).
    method_name : str
        Runtime method name invoked when the hook runs.
    method_kwargs : dict[str, Any]
        Default kwargs merged into hook invocation kwargs.
    init_params : dict[str, Any]
        Metadata-only declaration payload for class/type/library docs.
    """

    hook_name: str
    method_name: str
    method_kwargs: dict[str, Any] = field(default_factory=dict)
    init_params: dict[str, Any] = field(default_factory=dict)

    def declares_hook(self, hook_name: str) -> bool:
        return hook_name == self.hook_name

    def _invoke(self, runtime: "DataConfig", **kwargs):
        method = getattr(runtime, self.method_name, None)
        if not callable(method):
            raise AttributeError(
                f"Runtime '{type(runtime).__name__}' has no callable '{self.method_name}'",
            )
        call_kwargs = dict(self.method_kwargs)
        call_kwargs.update(kwargs)
        return method(**call_kwargs)

    def __call__(self, runtime: "DataConfig", *args, **kwargs):
        """Common plugin callable contract used across runtime adapters.

        Parameters
        ----------
        runtime : DataConfig
            Runtime data config instance orchestrating plugin hooks.
        *args : Any
            Positional runtime args (reserved for contract parity).
        **kwargs : Any
            Hook runtime kwargs. Optional ``hook_name`` filters execution.
        """
        _ = args
        hook_name = kwargs.pop("hook_name", None)
        if hook_name is not None and hook_name != self.hook_name:
            return None
        return self._invoke(runtime, **kwargs)

    def __getattr__(self, attr_name: str):
        if attr_name != self.hook_name:
            raise AttributeError(attr_name)

        def _hook(runtime: "DataConfig", *args, **kwargs):
            return self(runtime, *args, hook_name=attr_name, **kwargs)

        return _hook


@dataclass(eq=False, kw_only=True)
class DataConfig(ConfigBase):
    """
    Configuration and utility class for loading, preprocessing, and splitting datasets for machine learning tasks.

    Attributes
    -------
    dataset_name : str
        Name of the dataset to load or path to a data file.
    data_params : dict
        Additional parameters for data loading or generation.
    test_size : float
        Proportion of the dataset to include in the test split (between 0 and 1).
    train_size : float
        Proportion or count of samples to include in the training split.
    val_size : Union[float, int, None]
        Proportion or count of samples to include in the validation split when a
        ``sampler`` is provided (e.g. :class:`~deckard.data.sample.SplitSampler` or
        :class:`~deckard.data.sample.ShuffleSampler`).  Unused in legacy mode.
    split : Union[int, None]
        Which split index to use as the validation set when ``sampler`` performs
        cross-validation or shuffle splitting
        (e.g. :class:`~deckard.data.sample.KFoldSampler` or
        :class:`~deckard.data.sample.ShuffleSampler`).  Defaults to ``0``.
    sampler : Union[BaseSampler, Literal["split", "shuffle", "fold"], dict, None]
        Optional pluggable sampler spec. Resolution/configuration/execution is
        centralized in :class:`~deckard.data.sample.BaseSampler`. Accepted forms
        include alias strings (``split``/``shuffle``/``fold``), instantiated
        sampler objects, sampler classes, and Hydra-style dicts with
        ``name``/``_target_``.
    random_state : int
        Seed for random number generation to ensure reproducibility.
    stratify : Union[None, str, bool]
        Specifies stratification for sampling; can be None, True (use target), or a column name.
    classifier: bool
        Whether the task is classification (True) or regression (False).
    drop: list
        List of columns to drop from the dataset.
    target: Union[str, None]
        Name of the target column in the dataset (if applicable).
    keep: list
        List of columns to keep in the dataset.
    plugins : list
        Optional data plugin specifications executed during load/sample/score hooks.
    _X : pd.DataFrame
        Loaded feature matrix.
    _y : pd.Series
        Loaded target vector.
    data_load_time : float
        Time taken to load the data.
    data_sample_time : float
        Time taken to sample/split the data.
    train_n : int
        Number of training samples.
    test_n : int
        Number of testing samples.
    val_n : int
        Number of validation samples (set only when a sampler is used).
    alias: str
        Optional alias for the dataset configuration.
    train_indices : list
        Indices for training samples.
    test_indices : list
        Indices for testing samples.
    val_indices : list
        Indices for validation samples (set only when a sampler is used).
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target vector.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_test : pd.Series
        Testing target vector.
    X_val : pd.DataFrame
        Validation feature matrix (set only when a sampler is used).
    y_val : pd.Series
        Validation target vector (set only when a sampler is used).
    score_dict : dict
        Dictionary to store scores or metrics.
    _target_ : str
        Internal identifier for the class.

    Parameter layers
    ----------------
    data_params : dict
        Dataset loader/generator kwargs consumed by the selected data source
        (for example sklearn loaders, OpenML fetchers, synthetic generators,
        or file readers).
    plugins : list
        Plugin specs resolved at runtime and invoked through hook names.
        For :class:`DataHookPlugin`, initialization fields are:
        ``hook_name``, ``method_name``, ``method_kwargs``, and ``init_params``
        (metadata only).
    init_params (plugin metadata)
        Plugin declaration metadata (class/type/library docs). This is not
        interpreted by DataConfig orchestration logic directly.

    Family-specific parameter semantics
    ----------------------------------
    sklearn loader datasets
        ``data_params`` are forwarded to sklearn dataset loader callables
        (for example ``as_frame=True``).
    OpenML/fetch datasets
        ``data_params`` typically include source-identifying keys (for example
        ``name``, ``version``, ``as_frame``) and are passed to fetch helpers.
    synthetic generators
        ``data_params`` are generation controls (for example
        ``n_samples``, ``n_features``, ``n_classes``).
    file-backed datasets
        ``data_params`` may include read-time options used by pandas/file
        loading utilities.

    Plugin hook runtime params
    --------------------------
    Hooks are orchestrated by ``_run_plugin_hook(hook_name, **kwargs)``.
    Core hook names used by DataConfig runtime are:
    ``before_load_data``, ``after_load_data``, ``before_sample``,
    ``after_sample``, ``before_score``, and ``after_score``.
    Hook kwargs are phase-specific runtime objects supplied by the caller;
    DataHookPlugin forwards them to ``method_name`` after merging
    ``method_kwargs``.

    Methods
    -------
    __post_init__()
        Post-initialization method to validate parameters and initialize internal attributes.
    __hash__()
        Computes a hash value for the instance based on non-private attributes.
    _get_stratify_col()
        Returns the stratification array (or None) based on ``self.stratify``.
    load_dataset()
        Loads the dataset payload into runtime features/targets from defaults or file-backed datasets.
    sample()
        Splits the loaded dataset into training, testing, and optionally validation sets.
    score(*args, mode=None, **kwargs)
        Scores the selected data split via the configured data scorer.
    __call__(filepath=None)
        Loads and samples the dataset, splits it into training and testing sets, and returns the corresponding features and labels.
    save(filepath)
        Saves the current state of the DataConfig instance to a file.
    load(filepath=None)
        Loads a cached DataConfig object from pickle-based artifact storage.
    Raises
    ------
    ValueError
        For invalid parameter values or missing data.
    NotImplementedError
        For unsupported datasets or file types.

    Examples
    --------
    >>> config = DataConfig(dataset_name="adult", **kwargs)
    >>> config()
    >>> X_train = config.X_train
    >>> y_train = config.y_train
    >>> X_test = config.X_test
    >>> y_test = config.y_test
    >>> score_dict = config.score_dict

    Using a pluggable sampler for 3-way splits::

        from deckard.data.sample import SplitSampler
        config = DataConfig(
            dataset_name="digits",
            test_size=0.2,
            val_size=0.1,
            sampler=SplitSampler(),
        )
        config()
        X_val, y_val = config.X_val, config.y_val
    """

    # Configuration fields
    dataset_name: str = "adult"
    data_params: dict = None
    test_size: Union[float, int, None] = None
    train_size: Union[float, int, None] = None
    val_size: Union[float, int, None] = None
    split: Union[int, None] = None
    sampler: Union["BaseSampler", Literal["split", "shuffle", "fold"], dict, None] = "split"
    random_state: int = 42
    stratify: Union[None, str, bool] = True
    classifier: Union[bool, str] = True
    target: Union[str, None] = None
    drop: list = None
    keep: list = None
    plugins: list = field(default_factory=list)
    alias: Union[str, None] = None
    scorer: Any = AUTO_SCORER
    score_mode: str = "test"

    # Runtime state fields
    score_dict: dict = field(init=False, repr=True)
    data_load_time: Union[float, None] = None
    data_sample_time: Union[float, None] = None
    _X: Union[pd.DataFrame, pd.Series, None] = None
    _y: Union[pd.Series, None] = None
    train_indices: Union[list, None] = None
    test_indices: Union[list, None] = None
    val_indices: Union[list, None] = None
    X_train: Union[pd.DataFrame, pd.Series, None] = None
    y_train: Union[pd.Series, None] = None
    X_test: Union[pd.DataFrame, pd.Series, None] = None
    y_test: Union[pd.Series, None] = None
    X_val: Union[pd.DataFrame, pd.Series, None] = None
    y_val: Union[pd.Series, None] = None
    train_n: Union[int, None] = None
    test_n: Union[int, None] = None
    val_n: Union[int, None] = None
    _target_: Union[str, None] = None
    _plugin_objects: Union[list, None] = None
    _sampler_obj: Union[callable, None] = None

    def _validate_init(self):
        """
        Post-initialization method for setting up data-related attributes.

        Validates that `test_size` is between 0 and 1, then initializes training size and internal attributes
        for data loading, sampling, parameters, and train/test splits.

        Raises:
            ValueError: If `test_size` is not between 0 and 1.
        """
        if self.train_size is None:
            if self.test_size is None:
                self.test_size = 0.2
                self.train_size = 0.8
            else:
                if isinstance(self.test_size, float):
                    if not (0 < self.test_size < 1):
                        raise ValueError("test_size must be between 0 and 1")
                    self.train_size = 1 - self.test_size
                elif isinstance(self.test_size, int):
                    self.train_size = None
                else:
                    raise ValueError("test_size must be a float or int")
        self.data_params = self.data_params if self.data_params is not None else {}
        self.drop = [] if not hasattr(self, "drop") or self.drop is None else self.drop
        self.keep = [] if not hasattr(self, "keep") or self.keep is None else self.keep
        for attr in [
            "data_load_time",
            "data_sample_time",
            "_X",
            "_y",
            "train_indices",
            "test_indices",
            "val_indices",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "X_val",
            "y_val",
            "train_n",
            "test_n",
            "val_n",
            "split",
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, None)
        if not hasattr(self, "score_dict") or self.score_dict is None:
            self.score_dict = {}

        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.data.DataConfig"
        if not self.data_params:
            self.data_params = {}

    def _resolve_max_samples(self, dataset_len: int) -> Union[int, None]:
        """Resolve an optional dataset cap from the test-only environment variable."""
        max_samples = os.environ.get(DECKARD_TEST_MAX_SAMPLES_ENV)
        if max_samples in [None, ""]:
            return None
        try:
            max_samples = int(max_samples)
        except (TypeError, ValueError):
            raise ValueError(
                f"{DECKARD_TEST_MAX_SAMPLES_ENV} must be an integer, got {max_samples}",
            )
        if max_samples <= 0:
            return None
        return min(max_samples, dataset_len)

    def _apply_max_samples(self):
        """Truncate loaded tabular data to the configured test-only max sample cap."""
        if self._X is None or self._y is None:
            return

        sample_cap = self._resolve_max_samples(len(self._y))
        if sample_cap is None or sample_cap >= len(self._y):
            return

        if isinstance(self._X, pd.DataFrame):
            self._X = self._X.iloc[:sample_cap].copy()
        elif isinstance(self._X, pd.Series):
            self._X = self._X.iloc[:sample_cap].copy()
        else:
            raise TypeError(
                f"Unsupported _X type for {DECKARD_TEST_MAX_SAMPLES_ENV}: {type(self._X)}",
            )

        self._y = self._y.iloc[:sample_cap].copy()

    def _copy_runtime_state_to(self, target) -> None:
        runtime_fields = [
            "score_dict",
            "data_load_time",
            "data_sample_time",
            "_X",
            "_y",
            "train_indices",
            "test_indices",
            "val_indices",
            "X_train",
            "y_train",
            "X_test",
            "y_test",
            "X_val",
            "y_val",
            "train_n",
            "test_n",
            "val_n",
            "pipeline_fit_n",
            "pipeline_transform_n",
            "pipeline_fit_time",
            "pipeline_transform_time",
            "pipeline_y_fit_n",
            "pipeline_y_fit_time",
            "pipeline_y_transform_n",
            "pipeline_y_transform_time",
        ]
        for attr in runtime_fields:
            if hasattr(self, attr):
                setattr(target, attr, getattr(self, attr, None))

    def __post_init__(self):
        self._validate_init()
        self.scorer = _coerce_scorer_config(
            self.scorer,
            default_factory=lambda: load_class(
                (
                    "deckard.score.data.DefaultDataClassificationConfig"
                    if self.classifier
                    else "deckard.score.data.DefaultDataRegressionConfig"
                ),
            ),
        )
        if (
            self.scorer is not None
            and str(
                getattr(self.scorer, "scoring_type", ""),
            ).strip()
            == ""
        ):
            try:
                setattr(self.scorer, "scoring_type", "data")
            except Exception:
                pass
        if self.scorer is not None and not _is_data_scorer_instance(self.scorer):
            scoring_type = (
                str(getattr(self.scorer, "scoring_type", "")).strip().lower()
            )
            if scoring_type in {"model", "attack"}:
                raise TypeError(
                    "DataConfig requires a data-profile scorer configuration. "
                    "Model/attack scorers are not valid for data-split scoring.",
                )
        self._initialize_runtime_components()

    def _initialize_runtime_components(self) -> None:
        """Instantiate runtime-bound plugin and sampler objects eagerly."""
        from .sample import BaseSampler

        self._plugin_objects = self._get_plugins()
        self._sampler_obj = BaseSampler.resolve(self)

    @property
    def X(self) -> MatrixLike | None:
        """Convenience alias for the loaded feature matrix."""
        return self._X

    @property
    def y(self) -> ArrayLike | None:
        """Convenience alias for the loaded target vector."""
        return self._y

    @X.setter
    def X(self, value: MatrixLike | None) -> None:
        """Set the loaded feature matrix."""
        self._X = value

    @y.setter
    def y(self, value: ArrayLike | None) -> None:
        """Set the loaded target vector."""
        self._y = value

    def load(self, filepath: Union[str, None] = None):
        """Load a cached DataConfig object from pickle artifact storage.

        This method does not materialize datasets. Use :meth:`load_dataset` for
        runtime dataset loading from defaults/files.
        """
        if filepath is None:
            raise ValueError("filepath is required for DataConfig.load()")
        loaded = super().load(filepath)
        if isinstance(loaded, DataConfig) and loaded is not self:
            self.__dict__.update(loaded.__dict__)
            return self
        return loaded

    def save(self, payload: Any = None, filepath: str | None = None) -> None:
        """Save this DataConfig object as a pickle cache artifact."""
        target_path = filepath
        if target_path is None and isinstance(payload, (str, Path)):
            target_path = str(payload)
            payload = self
        if payload is None:
            payload = self
        if target_path is None:
            raise ValueError("filepath is required for DataConfig.save()")
        super().save(payload=payload, filepath=target_path)

    def load_raw_data(self) -> tuple[MatrixLike, ArrayLike]:
        """Compatibility alias for runtime dataset loading."""
        self.load_dataset()
        return self._X, self._y

    def load_dataset(self) -> "DataConfig":
        """Materialize runtime dataset payload into ``_X``/``_y``."""
        if not hasattr(self, "data_load_time") or self.data_load_time is None:
            self._load_dataset_runtime()
        return self

    def load_default_dataset(self, dataset_name: str, **loader_params: Any):
        """Public default dataset load entry-point delegated to declarations."""
        from .declarations import load_default_dataset

        return load_default_dataset(self, dataset_name=dataset_name, **loader_params)

    def split_data(self, run_hooks: bool = True) -> "DataConfig":
        """Compatibility alias for split-only behavior used by legacy call sites."""
        return self.fit(run_hooks=run_hooks)

    def fit(self, run_hooks: bool = True) -> "DataConfig":
        """Materialize train/test/(optional val) splits for this dataset."""
        self.load_dataset()
        if not hasattr(self, "data_sample_time") or self.data_sample_time is None:
            self._split_loaded_data(run_hooks=run_hooks)
        return self

    def sample(self, run_hooks: bool = True) -> "DataConfig":
        """Public sampling lifecycle method.

        Mirrors ``score/scorer`` naming with ``sample/sampler``.
        """
        self.fit(run_hooks=run_hooks)
        return self

    def score(self, *args, mode: str | None = None, **kwargs) -> dict:
        """Run data scoring for a canonical stage mode."""
        resolved_mode = mode if mode is not None else getattr(self, "score_mode", None)
        return self._score_runtime(*args, mode=resolved_mode, **kwargs)

    def _instantiate_plugin(self, plugin_spec: Any):
        return instantiate_plugin_spec(plugin_spec, loader=load_class)

    def _get_plugins(self) -> list:
        if not hasattr(self, "_plugin_objects") or self._plugin_objects is None:
            plugin_specs = normalize_plugin_specs(self.plugins)
            self._plugin_objects = [
                self._instantiate_plugin(spec) for spec in plugin_specs
            ]
        return self._plugin_objects

    def _run_plugin_hook(self, hook_name: str, **kwargs) -> list[Any]:
        """Execute one plugin hook across all instantiated plugins.

        Supported DataConfig hook names used by the runtime include:
        ``before_load_data``, ``after_load_data``, ``before_sample``,
        ``after_sample``, ``before_score``, and ``after_score``.
        """
        hook_outputs = []
        for plugin in self._get_plugins():
            hook = getattr(plugin, hook_name, None)
            if callable(hook):
                hook_outputs.append(hook(self, **kwargs))
        return hook_outputs

    
    def _run_score_stage_hooks(
        self,
        when: str,
        stage: str,
        **kwargs,
    ) -> list[Any]:
        """Run score hooks for a canonical stage with legacy compatibility.

        Hook dispatch order:
        1) stage-specific hook (e.g., ``after_score_test``)
        2) legacy generic hook (e.g., ``after_score``)
        """
        event = str(when).strip().lower()
        if event not in {"before", "after"}:
            raise ValueError(f"Score hook event must be 'before' or 'after', got {when}")
        stage_token = stage_hook_token(stage)
        stage_kwargs = {"stage": stage, **kwargs}
        outputs: list[Any] = []
        outputs.extend(
            self._run_plugin_hook(
                f"{event}_score_{stage_token}",
                **stage_kwargs,
            ),
        )
        outputs.extend(self._run_plugin_hook(f"{event}_score", **stage_kwargs))
        return outputs

    def _get_stratify_col(self, stratify: Union[None, str, bool] = None):
        """Return the stratification array (or ``None``) based on a stratify value.

        Returns
        -------
        pd.Series or None
            The column to stratify on, or ``None`` if stratification is disabled.

        Raises
        ------
        ValueError
            If ``stratify`` is a string that is not a column name in ``self._X``,
            or if ``stratify`` is an unrecognized type.
        """
        if stratify is None:
            stratify = getattr(self, "stratify", None)
        if stratify is None or stratify is False:
            return None
        if stratify is True:
            if self.classifier is False:
                return None
            return self._y
        if isinstance(stratify, str):
            if self._X is not None and stratify in self._X.columns:
                return self._X[stratify]
            raise ValueError(
                f"Stratify column '{stratify}' not found in data columns",
            )
        raise ValueError("stratify must be None, True, False, or a column name")

    def __hash__(self):
        return super().__hash__()

    def _encode_binary_series(
        self,
        series: pd.Series,
        mapping: dict[str, int],
    ) -> pd.Series:
        encoded = series.map(mapping)
        if encoded.isna().any():
            unique_values = [
                value for value in series.dropna().unique().tolist() if value != "nan"
            ]
            if len(unique_values) != 2:
                raise ValueError(
                    f"Expected a binary series, found values {sorted(unique_values)}",
                )
            fallback_mapping = {
                value: idx for idx, value in enumerate(sorted(unique_values))
            }
            encoded = series.map(fallback_mapping)
        return encoded.astype(int)

    def _split_loaded_data(
        self,
        run_hooks: bool = True,
    ):
        """
        Samples training, testing, and optionally validation indices from the loaded dataset.

        Delegates sampler resolution/configuration/execution to
        :class:`~deckard.data.sample.BaseSampler`, then materializes
        ``X_train``/``X_test`` and optional ``X_val`` splits.

        Raises
        ------
        ValueError
            If data is not loaded, or if the specified stratify column is not found, or if ``stratify`` is invalid.

        Side Effects
        ------------
        Sets ``self.train_indices``, ``self.test_indices``, ``self.val_indices`` (may be
        ``None``), and ``self.data_sample_time``.
        Logs the time taken for sampling.
        """
        if run_hooks:
            self._run_plugin_hook("before_sample")
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Cannot sample.")

        start_time = time.process_time()

        from .sample import BaseSampler

        train_idx, test_idx, val_idx = BaseSampler.execute(self)
        self.train_indices = train_idx
        self.test_indices = test_idx
        self.val_indices = val_idx
        if len(val_idx) > 0:
            self.X_val = self._X.iloc[self.val_indices].reset_index(drop=True)
            self.y_val = self._y.iloc[self.val_indices].reset_index(drop=True)
            self.val_n = len(self.X_val)

        end_time = time.process_time()
        self.data_sample_time = end_time - start_time
        logger.info(f"Data sampled in {self.data_sample_time:.2f} seconds")

        self.X_train = self._X.iloc[self.train_indices].reset_index(drop=True)
        self.y_train = self._y.iloc[self.train_indices].reset_index(drop=True)
        self.X_test = self._X.iloc[self.test_indices].reset_index(drop=True)
        self.y_test = self._y.iloc[self.test_indices].reset_index(drop=True)
        self.train_n = len(self.X_train)
        self.test_n = len(self.X_test)
        assert isinstance(
            self.X_train,
            (pd.DataFrame, pd.Series),
        ), "X_train must be a DataFrame"
        assert isinstance(self.y_train, pd.Series), "y_train must be a Series"
        assert isinstance(
            self.X_test,
            (pd.DataFrame, pd.Series),
        ), "X_test must be a DataFrame"
        assert isinstance(self.y_test, pd.Series), "y_test must be a Series"
        assert (
            hasattr(self, "train_indices") and self.train_indices is not None
        ), "Train indices must be set after sampling"
        assert (
            hasattr(self, "test_indices") and self.test_indices is not None
        ), "Test indices must be set after sampling"
        assert isinstance(
            self.X_train,
            (pd.DataFrame, pd.Series),
        ), f"X_train must be a DataFrame or Series, got {type(self.X_train)}"
        assert isinstance(
            self.y_train,
            pd.Series,
        ), f"y_train must be a Series, got {type(self.y_train)}"
        assert isinstance(
            self.X_test,
            (pd.DataFrame, pd.Series),
        ), f"X_test must be a DataFrame or Series, got {type(self.X_test)}"
        assert isinstance(
            self.y_test,
            pd.Series,
        ), f"y_test must be a Series, got {type(self.y_test)}"
        if run_hooks:
            self._run_plugin_hook("after_sample")

    def _load_dataset_runtime(self):
        """
        Loads dataset based on the provided dataset name or file type.

        Supported datasets
        ------------------
        - "adult"
        - "make_classification"
        - "make_regression"
        - "diabetes"
        - "digits"
        - "iris"

        Supported file types
        --------------------
        - ".csv" (must contain a 'target' column)

        For built-in datasets, calls the corresponding loader method.
        For CSV files, reads the file and splits features and target.
        Raises NotImplementedError for unsupported datasets or file types.
        Updates ``self._X``, ``self._y``, and ``self.data_load_time`` with loaded data and timing information.

        Raises
        ------
        NotImplementedError
            If the dataset or file type is not supported.
        ValueError
            If a CSV file does not contain a 'target' column.
        """
        if hasattr(self, "data_load_time") and self.data_load_time is not None:
            return
        self._run_plugin_hook("before_load_data")
        from .declarations import build_loader_registry

        supported_datasets = build_loader_registry(self)
        filetype = Path(self.dataset_name).suffix
        supported_filetypes = data_supported_filetypes
        if (
            filetype not in supported_filetypes
            and self.dataset_name not in supported_datasets
        ):
            raise NotImplementedError(
                f"Currently only {supported_filetypes} filetypes are supported for loading data. Cannot load {self.dataset_name}",
            )
        if self.dataset_name in supported_datasets:
            start_time = time.process_time()
            self.load_default_dataset(self.dataset_name, **self.data_params)
        elif filetype == ".openml":
            start_time = time.process_time()
            dataset_base_name = Path(self.dataset_name).stem
            from .declarations import load_generic_openml

            load_generic_openml(
                self,
                dataset_name=dataset_base_name,
                **self.data_params,
            )
        elif filetype in supported_filetypes:
            start_time = time.process_time()
            self._load_from_csv(**self.data_params)
            end_time = time.process_time()
            self.data_load_time = end_time - start_time
        else:
            raise NotImplementedError(
                f"Dataset {self.dataset_name} not implemented",
            )

        assert isinstance(
            self._X,
            (pd.DataFrame, pd.Series),
        ), "_X must be a DataFrame after loading data"
        assert isinstance(
            self._y,
            pd.Series,
        ), "_y must be a Series after loading data"
        self._apply_max_samples()
        self._run_plugin_hook("after_load_data")
        logger.info(
            f"Data loaded from {self.dataset_name} in {self.data_load_time:.2f} seconds",
        )

    def _load_from_csv(self):
        data = self.load_data(self.dataset_name)
        if self.target is None:
            raise ValueError(
                "CSV file must contain a 'target' column or specify the target column name in the 'target' attribute",
            )
        y = data.pop(self.target)
        if len(self.keep) > 1:
            data = data[self.keep]
        elif len(self.keep) == 1:
            data = data[self.keep[0]]
        for del_col in self.drop:
            assert len(self.keep) == 0, "Cannot specify both keep and drop columns"
            if del_col in data.columns:
                data = data.drop(columns=del_col)
        self._X = data
        self._y = y

    def _score_runtime(
        self,
        *args,
        mode: str | None = None,
        **kwargs,
    ) -> dict:
        """
        Delegates all dataset scoring to ``self.scorer``. Supports pre-sample mode (raw data, only in DataConfig),
        as well as train/test/val splits. If mode is not provided, uses self.score_mode or defaults to 'test'.
        """
        if self.scorer is None:
            return {}
        if not callable(self.scorer):
            raise TypeError(
                f"DataConfig.scorer must be callable or None, got {type(self.scorer)}",
            )
        if not _is_data_scorer_instance(self.scorer):
            raise TypeError(
                "DataConfig.scorer must be a data-profile scorer; "
                "model/attack scorers cannot run on raw X/y data splits.",
            )
        scorer_mode = normalize_data_score_stage(
            mode or getattr(self, "score_mode", None) or "test",
        )
        self._run_score_stage_hooks("before", scorer_mode)
        if scorer_mode == "pre-sample":
            y_true = getattr(self, "_y", None)
            y_pred = getattr(self, "_X", None)
        elif scorer_mode == "train":
            y_true = getattr(self, "y_train", None)
            y_pred = getattr(self, "X_train", None)
        elif scorer_mode == "test":
            y_true = getattr(self, "y_test", None)
            y_pred = getattr(self, "X_test", None)
        elif scorer_mode == "val":
            y_true = getattr(self, "y_val", None)
            y_pred = getattr(self, "X_val", None)
        elif scorer_mode in {"post-sample", "post-pipeline", "all"}:
            # TODO: Ensure that post-sample and post-pipeline are handled in pipeline stage correctly.
            y_train = getattr(self, "y_train", None)
            y_test = getattr(self, "y_test", None)
            X_train = getattr(self, "X_train", None)
            X_test = getattr(self, "X_test", None)
            if y_train is None or y_test is None or X_train is None or X_test is None:
                raise ValueError(
                    "Data scoring mode 'post-sample' requires both train and test splits.",
                )
            if isinstance(X_train, (pd.DataFrame, pd.Series)):
                y_pred = pd.concat([X_train, X_test], ignore_index=True)
            else:
                y_pred = np.concatenate([np.asarray(X_train), np.asarray(X_test)])
            if isinstance(y_train, (pd.DataFrame, pd.Series)):
                y_true = pd.concat([y_train, y_test], ignore_index=True)
            else:
                y_true = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
        else:
            raise ValueError(f"Mode must be one of canonical data score stages, got {scorer_mode}")
        if y_true is None or y_pred is None:
            raise ValueError(
                f"Data scoring mode '{scorer_mode}' requested but required data split is unavailable.",
            )
        result_dict = self.scorer(
            *args,
            y_true=y_true,
            y_pred=y_pred,
            mode=scorer_mode,
            data=self,
            **kwargs,
        )
        plugin_scores = self._run_score_stage_hooks(
            "after",
            scorer_mode,
            scores=result_dict,
        )
        for plugin_score in plugin_scores:
            if isinstance(plugin_score, dict):
                result_dict.update(plugin_score)
        return result_dict

    def apply_pipeline(self, pipeline) -> "DataConfig":
        """Attach a pipeline-like plugin object to this data config."""
        if pipeline is None:
            return self
        pipeline_plugins = (
            [pipeline] if not isinstance(pipeline, list) else list(pipeline)
        )
        existing_plugins = list(self.plugins or [])
        self.plugins = [*pipeline_plugins, *existing_plugins]
        return self

    def _prepare_data_file(self, data_file: Union[str, None]) -> bool:
        """
        Handles loading/saving behavior for data_file.

        Returns
        -------
        Tuple[DataConfig, bool]
            (possibly loaded config instance, save_flag)
        """
        if data_file is not None:
            data_path = Path(data_file)
            if data_path.exists():
                logger.info(f"Loading existing DataConfig from {data_file}")
                self.load(data_file)
                return False
            logger.debug(f"Creating directory for DataConfig at {data_file}")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            return True

        logger.debug("No data_file provided, data will not be saved")
        return False

    def prepare_data_file(self, data_file: str | None) -> bool:
        """Public wrapper for data-file load/save preparation behavior."""
        return self._prepare_data_file(data_file=data_file)

    def ensure_data_loaded(self) -> None:
        """Ensure raw data is loaded for downstream orchestration."""
        self.load_dataset()

    def ensure_data_sampled(self) -> None:
        """Ensure train/test/(optional val) splits are materialized."""
        self.fit()

    def build_data_time_dict(self) -> dict:
        """Build timing/count metadata dictionary for data runtime outputs."""
        time_dict = {"data_load_time": self.data_load_time}
        time_dict["data_sample_time"] = self.data_sample_time
        time_dict["train_n"] = self.train_n
        time_dict["test_n"] = self.test_n
        if self.val_n is not None:
            time_dict["val_n"] = self.val_n
        return time_dict

    def __call__(
        self,
        *args,
        data_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
        **kwargs,
    ) -> dict:
        """
        Loads and samples the dataset, splits it into training and testing sets, and returns timing and scoring information.
        Strictly validates that all output values are flat and serializable.
        """

        save_flag = self.prepare_data_file(data_file=data_file)
        scores = dict(getattr(self, "score_dict", {}) or {})
        self.load_dataset()
        logger.info(f"Data loaded in {self.data_load_time:.2f} seconds")
        self.fit()
        time_dict = self.build_data_time_dict()
        if self.X_val is not None:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}, "
                f"Val set size: {len(self.X_val)}",
            )
        else:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
            )
        data_scores = self.score(*args, **kwargs)
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = all_scores
        assert hasattr(self, "score_dict"), "score_dict must be set"
        all_scores = self.merge_and_persist_scores(all_scores, score_file)
        if save_flag:
            self.save(data_file)
        return self.score_dict


@dataclass
class DataPipelineStep:
    """
    Represents a step in a data pipeline with optional metadata.

    This dataclass normalizes and documents the optional parameters that
    configure how a step integrates with the pipeline runtime.

    Attributes
    ----------
    name : str
        Fully-qualified class name or file path (e.g., "sklearn.preprocessing.StandardScaler"
        or "path/to/custom.py:CustomTransformer").
    fit_y : bool, default=False
        If True, this step is applied to the target (y) rather than features (X).
        Only one of fit_y or fit_xy can be True.
    dtype : str, optional
        Column selector hint for ColumnTransformer. Supported values: "numeric",
        "num", "float", "int", "object", "string", "category". When specified,
        only columns of matching dtype are passed to this step.
    plugin_hook : str | list[str], optional
        Hook name(s) that trigger when this step runs. Supported hooks:
        - "before_sample": Run this step's pre_sample_fit before KFold sampling.
    """

    name: str
    fit_y: bool = False
    dtype: Optional[str] = None
    plugin_hook: Union[str, list, None] = None

    @classmethod
    def from_config(cls, step_name: str, step_config: dict) -> "DataPipelineStep":
        """
        Extract DataPipelineStep from a pipeline step config dict.

        Parameters
        ----------
        step_name : str
            Name of the step (key in pipeline.steps dict).
        step_config : dict
            Configuration dict for the step.

        Returns
        -------
        DataPipelineStep
            Parsed step metadata.

        Raises
        ------
        ValueError
            If "name" key is missing or if both fit_y and fit_xy are True.
        """
        step_class = step_config.get("name")
        if step_class is None:
            raise ValueError(f"Step {step_name} missing required 'name' key")

        fit_y = step_config.get("fit_y", False)
        fit_xy = step_config.get("fit_xy", False)

        if fit_xy is True:
            raise ValueError("fit_xy pipeline steps are no longer supported.")

        return cls(
            name=step_class,
            fit_y=fit_y,
            dtype=step_config.get("dtype", None),
            plugin_hook=step_config.get("plugin_hook", None),
        )

    def stripped_config(self, step_config: dict) -> dict:
        """
        Remove DataPipelineStep metadata from a step config dict.

        Parameters
        ----------
        step_config : dict
            Original step configuration.

        Returns
        -------
        dict
            New dict with metadata keys removed, ready for transformer instantiation.
        """
        config = dict(step_config)
        for key in {"name", "fit_y", "fit_xy", "dtype", "plugin_hook"}:
            config.pop(key, None)
        return config


@dataclass(eq=False, kw_only=True)
class DataPipelineConfig(DataPipelineMixin, DataConfig):
    """Initializes a data pipeline configuration and fits it to the data in the call() method."""

    pipeline: dict = field(default_factory=dict)
    pre_sample_transform: bool = False

    def __post_init__(self):
        self._validate_init()
        # Allow a list of step-dicts: merge them in order (later wins on key conflict)

        if isinstance(self.pipeline, (list, ListConfig)):
            self.pipeline = merge_list_of_dicts(coerce_to_list(self.pipeline))
        self._pipeline_step_hooks = {}
        assert isinstance(
            self.pipeline,
            (dict, DictConfig),
        ), f"pipeline must be a dictionary, got {type(self.pipeline)}"
        for attr in [
            "pipeline_fit_n",
            "pipeline_transform_n",
            "pipeline_fit_time",
            "pipeline_transform_time",
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, None)
        # Validate the pipeline configuration
        for k, v in self.pipeline.items():
            assert isinstance(
                v,
                (dict, DictConfig),
            ), f"Each step in pipeline must be a dictionary, got {type(v)} for step {k}"
            # Validate step using DataPipelineStep.from_config (also validates "name" is present)
            try:
                DataPipelineStep.from_config(k, v)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid pipeline step '{k}': {e}") from e
        if self.classifier in ["classifier", True]:
            self.classifier = True
        elif self.classifier in ["regressor", False]:
            self.classifier = False
        else:
            raise ValueError(
                f"classifier must be boolean or one of ['classifier', 'regressor'], got {self.classifier}",
            )
        self.scorer = _coerce_scorer_config(
            self.scorer,
            default_factory=lambda: load_class(
                (
                    "deckard.score.data.DefaultDataClassificationConfig"
                    if self.classifier
                    else "deckard.score.data.DefaultDataRegressionConfig"
                ),
            ),
        )
        self._initialize_runtime_components()

    def fit(self, run_hooks: bool = True) -> "DataPipelineConfig":
        """Load, sample, and apply configured pipeline transforms."""
        self.load_dataset()
        if not hasattr(self, "data_sample_time") or self.data_sample_time is None:
            self.run_sampling_with_pipeline_hooks()
        self.apply_pipeline_behavior()
        return self

    def _normalize_step_hooks(self, raw_hooks: Any) -> list[str]:
        if raw_hooks is None:
            return []
        if isinstance(raw_hooks, str):
            hooks = [raw_hooks]
        elif isinstance(raw_hooks, (list, tuple, set)):
            hooks = list(raw_hooks)
        else:
            raise TypeError(
                f"plugin_hook must be None, str, or list-like. Got {type(raw_hooks)}",
            )

        normalized = []
        for hook in hooks:
            text = str(hook).strip().lower()
            if text:
                normalized.append(text)
        return normalized

    def _pipeline_declares_hook(self, hook_name: str) -> bool:
        target_hook = str(hook_name).strip().lower()
        for _, step_config in self.pipeline.items():
            hooks = self._normalize_step_hooks(step_config.get("plugin_hook", None))
            if target_hook in hooks:
                return True
        return False

    def _resolve_step_config(self, step_class: str, step_config: dict) -> dict:
        """Resolve fold-specific paths in step config for precomputed matrices.

        For StringDistanceTransformer, resolves ${data.split} placeholders in
        distance_matrix_train and distance_matrix_test paths using self.split.
        """
        resolved_config = {**step_config}

        # Only process StringDistanceTransformer configs
        if "StringDistanceTransformer" not in step_class:
            return resolved_config

        # Resolve distance_matrix_full path if present
        if "distance_matrix_full" in resolved_config:
            path = resolved_config["distance_matrix_full"]
            if isinstance(path, str) and "${data.split}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_full"] = path.replace(
                        "${data.split}",
                        str(self.split),
                    )
            elif isinstance(path, str) and "${item.fold}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_full"] = path.replace(
                        "${item.fold}",
                        str(self.split),
                    )

        # Resolve distance_matrix_train path if present
        if "distance_matrix_train" in resolved_config:
            path = resolved_config["distance_matrix_train"]
            if isinstance(path, str) and "${data.split}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_train"] = path.replace(
                        "${data.split}",
                        str(self.split),
                    )
            elif isinstance(path, str) and "${item.fold}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_train"] = path.replace(
                        "${item.fold}",
                        str(self.split),
                    )

        # Resolve distance_matrix_test path if present
        if "distance_matrix_test" in resolved_config:
            path = resolved_config["distance_matrix_test"]
            if isinstance(path, str) and "${data.split}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_test"] = path.replace(
                        "${data.split}",
                        str(self.split),
                    )
            elif isinstance(path, str) and "${item.fold}" in path:
                if self.split is not None:
                    resolved_config["distance_matrix_test"] = path.replace(
                        "${item.fold}",
                        str(self.split),
                    )

        return resolved_config

    def _run_pre_sample_transformations(self, pipeline: Pipeline, force: bool = False):
        if not (self.pre_sample_transform or force):
            return
        # Compute budget: cap the data passed to pre_sample_fit to the configured split sizes
        train_size = getattr(self, "train_size", None) or 0
        test_size = getattr(self, "test_size", None) or 0
        val_size = getattr(self, "val_size", None) or 0
        # Only cap when sizes are integers (fractional sizes are not capped)
        if (
            isinstance(train_size, int)
            and isinstance(test_size, int)
            and isinstance(val_size, int)
        ):
            budget = train_size + test_size + val_size
        else:
            budget = 0
        n_samples = len(self._y) if self._y is not None else 0
        if budget > 0 and n_samples > budget:
            rng = np.random.default_rng(getattr(self, "random_state", None) or 0)
            keep_idx = np.sort(rng.choice(n_samples, size=budget, replace=False))
            if hasattr(self._X, "iloc"):
                X_hook = self._X.iloc[keep_idx].reset_index(drop=True)
            else:
                X_hook = self._X[keep_idx]
            if hasattr(self._y, "iloc"):
                y_hook = self._y.iloc[keep_idx].reset_index(drop=True)
            else:
                y_hook = self._y[keep_idx]
            logger.info(
                f"Subsetting data from {n_samples} to {budget} samples before pre_sample_fit.",
            )
        else:
            X_hook = self._X
            y_hook = self._y
        for step_name, step in pipeline.steps:
            configured_hooks = self._pipeline_step_hooks.get(step_name, [])
            if configured_hooks and "before_sample" not in configured_hooks:
                continue
            pre_sample_fit = getattr(step, "pre_sample_fit", None)
            if callable(pre_sample_fit):
                pre_sample_fit(X_hook, y=y_hook, data=self)

    def _inject_sample_indices(self, pipeline: Pipeline):
        if not hasattr(self, "train_indices") or self.train_indices is None:
            return
        for _, step in pipeline.steps:
            set_split_indices = getattr(step, "set_split_indices", None)
            if callable(set_split_indices):
                set_split_indices(
                    train_indices=self.train_indices,
                    test_indices=self.test_indices,
                    val_indices=self.val_indices,
                )

    def _init_pipeline(self):
        if not isinstance(self.pipeline, (dict, DictConfig)):
            raise ValueError(f"Invalid pipeline configuration: {self.pipeline}")
        self._pipeline_step_hooks = {}
        X_pipeline_steps = []
        y_pipeline_steps = []
        dtypes = []
        for step_name, step_config in self.pipeline.items():
            # Parse step metadata (name, fit_y, dtype, plugin_hook)
            step = DataPipelineStep.from_config(step_name, step_config)

            # Register hooks if present
            step_hooks = self._normalize_step_hooks(step.plugin_hook)
            if step_hooks:
                self._pipeline_step_hooks[step_name] = step_hooks

            # Get clean config (metadata removed, ready for instantiation)
            step_config_clean = step.stripped_config(step_config)

            # Resolve fold-specific paths before instantiation
            step_config_clean = self._resolve_step_config(step.name, step_config_clean)

            # Instantiate the transformer
            step_instance = load_class(step.name, **step_config_clean)
            dtypes.append(step.dtype)

            if step.fit_y is not True:
                X_pipeline_steps.append((step_name, step_instance))
            else:
                y_pipeline_steps.append((step_name, step_instance))

        if dtypes is not None and any(x is not None for x in dtypes):

            string_dtypes = {"object", "string", "category"}
            num_dtypes = {"num", "numeric", "float", "int"}

            transformers = []

            for (name, transformer), dtype in zip(X_pipeline_steps, dtypes):

                if dtype in num_dtypes:
                    selector = make_column_selector(dtype_include=np.number)

                elif dtype in string_dtypes:
                    selector = make_column_selector(
                        dtype_include=object,
                    )  # or "string" depending on your data

                else:
                    continue  # skip unknown dtype

                transformers.append((name, transformer, selector))

            X_pipeline = Pipeline(
                steps=[
                    (
                        "preprocess",
                        ColumnTransformer(
                            transformers=transformers,
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                        ),
                    ),
                ],
            )

        else:
            X_pipeline = Pipeline(steps=X_pipeline_steps)
        if len(y_pipeline_steps) > 0:
            y_pipeline = y_pipeline_steps
        else:
            y_pipeline = None
        return X_pipeline, y_pipeline

    def compose_pipeline_behavior(self):
        """Compose feature/target pipeline runtime behavior as needed for this config."""
        return self.create_pipeline()

    def should_run_pre_sample_pipeline(self) -> bool:
        """Determine whether pre-sample pipeline behavior is composed for this run."""
        return self.pre_sample_transform or self._pipeline_declares_hook(
            "before_sample",
        )

    def run_sampling_with_pipeline_hooks(self) -> None:
        """Compose sampling behavior with optional pre-sample pipeline hooks."""
        run_before_sample_pipeline = self.should_run_pre_sample_pipeline()
        if run_before_sample_pipeline:
            self._run_plugin_hook("before_sample")
            pre_sample_pipeline, _ = self.compose_pipeline_behavior()
            self._run_pre_sample_transformations(
                pre_sample_pipeline,
                force=run_before_sample_pipeline,
            )
            self._split_loaded_data(run_hooks=False)
            self._run_plugin_hook("after_sample")
            return
        self._split_loaded_data()

    def _transform_validation_with_pipeline(self, X_pipeline: Pipeline) -> None:
        """Apply the fitted feature pipeline to validation inputs when present."""
        if getattr(self, "X_val", None) is None or not X_pipeline.steps:
            return
        if isinstance(self.X_val, pd.DataFrame):
            val_cols = self.X_val.columns
        elif isinstance(self.X_val, pd.Series):
            val_cols = [self.X_val.name]
        else:
            val_cols = [f"feature_{i}" for i in range(self.X_val.shape[1])]
        X_val_transformed = X_pipeline.transform(self.X_val)
        from scipy.sparse import issparse

        if issparse(X_val_transformed):
            X_val_transformed = X_val_transformed.toarray()
        try:
            val_cols_out = list(X_pipeline.get_feature_names_out(val_cols))
        except AttributeError:
            n_features = X_val_transformed.shape[1]
            val_cols_out = (
                list(val_cols)
                if len(val_cols) == n_features
                else [f"feature_{i}" for i in range(n_features)]
            )
        self.X_val = pd.DataFrame(X_val_transformed, columns=val_cols_out)

    def apply_pipeline_behavior(self) -> None:
        """Compose and apply feature/target pipeline behavior to sampled splits."""
        X_pipeline, y_pipeline = self.compose_pipeline_behavior()
        self._inject_sample_indices(X_pipeline)
        self.X_train, self.X_test, self.y_train, self.y_test = self._fit_transform_X(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            X_pipeline,
        )
        self._transform_validation_with_pipeline(X_pipeline)
        if y_pipeline is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self._fit_transform_y(
                    self.X_train,
                    self.X_test,
                    self.y_train,
                    self.y_test,
                    y_pipeline,
                )
            )

    def create_pipeline(self):
        """Public entry-point for pipeline initialisation. Delegates to `_init_pipeline()`."""
        return self._init_pipeline()

    def _fit_transform_X(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        pipeline,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fits the data pipeline to the data and returns the transformed data.

        Parameters
        ----------
        data : DataConfig
            The data configuration object containing the training and testing data.

        Returns
        -------
        pd.DataFrame
            The transformed training and testing data.
        """

        def _resolve_feature_names(input_cols, transformed_X):
            try:
                return list(pipeline.get_feature_names_out(input_cols))
            except AttributeError:
                n_features = transformed_X.shape[1]
                if len(input_cols) == n_features:
                    return list(input_cols)
                return [f"feature_{i}" for i in range(n_features)]

        # Nothing to do for an empty pipeline — skip fit/transform entirely.
        if not pipeline.steps:
            self.pipeline_fit_time = 0.0
            self.pipeline_fit_n = X_train.shape[0]
            self.pipeline_transform_time = 0.0
            self.pipeline_transform_n = X_test.shape[0]
            return X_train, X_test, y_train, y_test

        if not hasattr(self, "pipeline_fit_time") or self.pipeline_fit_time is None:
            logger.info("Fitting data pipeline to training data")
            # Fit and transform the training data
            start = time.process_time()
            pipeline.fit(X_train, y_train)
            end = time.process_time()
            before_shape = X_train.shape
            if isinstance(X_train, pd.DataFrame):
                train_cols = X_train.columns
            elif isinstance(X_train, pd.Series):
                train_cols = [X_train.name]
            else:
                train_cols = [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train = pipeline.transform(X_train)
            # If csr_matrix, convert to dense
            if isinstance(X_train, csr_matrix):
                X_train = X_train.toarray()
            train_cols = _resolve_feature_names(train_cols, X_train)
            X_train = pd.DataFrame(X_train, columns=train_cols)
            after_shape = X_train.shape
            assert (
                before_shape[0] == after_shape[0]
            ), f"Number of samples changed during fit_transform from {before_shape[0]} to {after_shape[0]}"
            self.pipeline_fit_time = end - start
            self.pipeline_fit_n = X_train.shape[0]
        if (
            not hasattr(self, "pipeline_transform_time")
            or self.pipeline_transform_time is None
        ):
            # Record transform time
            start = time.process_time()
            # Transform the testing data
            before_shape = X_test.shape
            if isinstance(X_test, pd.DataFrame):
                test_cols = X_test.columns
            elif isinstance(X_test, pd.Series):
                test_cols = [X_test.name]
            else:
                test_cols = [f"feature_{i}" for i in range(X_test.shape[1])]
            X_test = pipeline.transform(X_test)
            if isinstance(X_test, csr_matrix):
                X_test = X_test.toarray()
            test_cols = _resolve_feature_names(test_cols, X_test)
            # Ensure transformed data is a DataFrame with correct columns
            X_test = pd.DataFrame(X_test, columns=test_cols)
            after_shape = X_test.shape
            assert (
                before_shape[0] == after_shape[0]
            ), f"Number of samples changed during transform from {before_shape[0]} to {after_shape[0]}"
            end = time.process_time()
            self.pipeline_transform_time = end - start
            self.pipeline_transform_n = X_test.shape[0]
        return X_train, X_test, y_train, y_test

    def _fit_transform_y(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        pipeline,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fits the data pipeline to y_train and transforms y_train/y_test."""

        # Normalize y to 2D for sklearn transformers
        y_train_2d = (
            y_train.to_frame()
            if isinstance(y_train, pd.Series)
            else pd.DataFrame(y_train)
        )
        y_test_2d = (
            y_test.to_frame()
            if isinstance(y_test, pd.Series)
            else pd.DataFrame(y_test)
        )

        if (
            not hasattr(self, "pipeline_y_fit_time")
            or self.pipeline_y_fit_time is None
        ):
            logger.info("Fitting data pipeline to training target")
            start = time.process_time()
            for name, stage in pipeline:
                logger.debug(f"Running data pipeline stage: {name}")
                stage.fit(y_train_2d)
                y_train_t = stage.transform(y_train_2d)
            end = time.process_time()

            before_shape = y_train_2d.shape
            y_train_t = stage.transform(y_train_2d)
            if isinstance(y_train_t, csr_matrix):
                y_train_t = y_train_t.toarray()
                y_train_t = pd.DataFrame(y_train_t)
                after_shape = y_train_t.shape

                assert (
                    before_shape[0] == after_shape[0]
                ), f"Number of samples changed during y fit_transform from {before_shape[0]} to {after_shape[0]}"

            self.pipeline_y_fit_time = end - start
            self.pipeline_y_fit_n = y_train_t.shape[0]
            y_train = pd.Series(y_train_t)

        if (
            not hasattr(self, "pipeline_y_transform_time")
            or self.pipeline_y_transform_time is None
        ):
            start = time.process_time()

            before_shape = y_test_2d.shape
            y_test_t = stage.transform(y_test_2d)
            if isinstance(y_test_t, csr_matrix):
                y_test_t = y_test_t.toarray()
                y_test_t = pd.DataFrame(y_test_t)
                after_shape = y_test_t.shape

                assert (
                    before_shape[0] == after_shape[0]
                ), f"Number of samples changed during y transform from {before_shape[0]} to {after_shape[0]}"

            end = time.process_time()
            self.pipeline_y_transform_time = end - start
            self.pipeline_y_transform_n = y_test_t.shape[0]
            y_test = pd.Series(y_test_t)

        return X_train, X_test, y_train, y_test

    def __call__(
        self,
        *args,
        data_file: Union[str, None] = None,
        score_file: Union[str, None] = None,
        mode: str = None,
        **kwargs,
    ) -> dict:
        """
        Loads and samples the dataset, splits it into training and testing sets, and returns timing and scoring information.
        Parameters
        ----------
        data_file : Union[str, None]
            Path to save loaded data as CSV. If None, data is not saved.
        score_file : Union[str, None]
            Path to save scores as CSV. If None, scores are not saved.
        Returns
        -------
        dict:
            A dictionary containing:
            - 'data_load_time': Time taken to load the data.
            - 'data_sample_time': Time taken to sample/split the data.
            - Additional times/scores can be added in the future.

        Raises
        ------
        AssertionError
            If train or test indices are not set after sampling.
        """
        save_flag = self._prepare_data_file(data_file=data_file)
        scores = self.read_or_initialize_scores(score_file)
        self.load_dataset()
        logger.info(f"Data loaded in {self.data_load_time:.2f} seconds")
        self.sample()
        time_dict = {
            "data_load_time": self.data_load_time,
            "data_sample_time": self.data_sample_time,
            "pipeline_fit_time": self.pipeline_fit_time,
            "pipeline_fit_n": self.pipeline_fit_n,
            "pipeline_transform_time": self.pipeline_transform_time,
            "pipeline_transform_n": self.pipeline_transform_n,
        }
        if getattr(self, "val_n", None) is not None:
            time_dict["val_n"] = self.val_n
        if getattr(self, "X_val", None) is not None:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}, "
                f"Val set size: {len(self.X_val)}",
            )
        else:
            logger.info(
                f"Train set size: {len(self.X_train)}, Test set size: {len(self.X_test)}",
            )
        self.score_dict.update(**time_dict)
        # Respect explicit mode first, then configured score_mode, then _score fallback.
        resolved_mode = mode if mode is not None else getattr(self, "score_mode", None)
        data_scores = self.score(*args, mode=resolved_mode, **kwargs)
        all_scores = {**scores, **data_scores, **time_dict}
        self.score_dict = all_scores
        assert hasattr(self, "score_dict"), "score_dict must be set"

        if score_file is not None and not Path(score_file).exists():
            self.save_scores(all_scores, score_file)
        if save_flag:
            self.save(data_file)
        return self.score_dict
