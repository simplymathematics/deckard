# Imports
import pandas as pd
import time
import logging
import inspect

from dataclasses import dataclass, field
from typing import Union, Literal, Optional, Any, Dict

# Scikit-learn
from sklearn.model_selection import (
    KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit, GroupKFold, LeaveOneGroupOut, LeavePGroupsOut
)

from scipy.sparse import csr_matrix



# deckard
from ..utils import ConfigBase
from ..data import DataConfig

supported_splitters = [
    "kfold",
    "shuffle_split",
    "stratified_kfold",
    "stratified_shuffle_split",
    "time_series_split",
    "group_kfold",
    "leave_one_group_out",
    "leave_p_groups_out",
]
logger = logging.getLogger(__name__)

@dataclass
class DataSplitterConfig(DataConfig):
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    type: Literal[f"{supported_splitters}"] = "kfold"
    stratified: bool = True
    splitter_params: Dict[str, Any] = field(default_factory=dict)
    split: int = 0  # Current split index
    
    
    def __init__(self, *args, **kwargs):
        # Inspect DataConfig __init__ signature to determine which arguments to pass
        signature = inspect.signature(DataConfig.__init__)
        valid_args = {k: v.default for k, v in signature.parameters.items() if k != 'self' and v.default is not inspect.Parameter.empty}
        # Filter kwargs to only include valid arguments for DataConfig
        data_config_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        # Initialize DataConfig part
        super().__init__(*args, **data_config_kwargs)
        # Now initialize SplitterConfig specific attributes
        for k, v in kwargs.items():
            if k not in data_config_kwargs:
                setattr(self, k, v)
    
    def get_params(self) -> Dict[str, Any]:
        # Find init params from self and parent classes
        params = {}
        for cls in self.__class__.mro():
            if hasattr(cls, '__init__'):
                signature = inspect.signature(cls.__init__)
                for k in signature.parameters.keys():
                    if k != 'self' and hasattr(self, k):
                        params[k] = getattr(self, k)
            
        return params
    
    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.__post_init__()

    def __post_init__(self):
        assert self.split < self.n_splits, "split index must be less than n_splits"
        if self.split < 0:
            raise ValueError("split index must be ")
        return super().__post_init__()
        
        
    
    def _initialize_splitter(self, y: Optional[Union[pd.Series, csr_matrix]] = None):
        if self.type == "kfold":
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
                **self.splitter_params,
            )
        elif self.type == "stratified_kfold":
            if y is None:
                raise ValueError("y must be provided for stratified_kfold")
            if self.stratified is False:
                raise ValueError("stratified must be True for stratified_kfold")
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
                **self.splitter_params,
            )
        elif self.type == "shuffle_split":
            splitter = ShuffleSplit(
                n_splits=self.n_splits,
                random_state=self.random_state,
                **self.splitter_params,
            )
        elif self.type == "stratified_shuffle_split":
            if y is None:
                raise ValueError("y must be provided for stratified_shuffle_split")
            if self.stratified is False:
                raise ValueError("stratified must be True for stratified_shuffle_split")
            splitter = StratifiedShuffleSplit(
                n_splits=self.n_splits,
                random_state=self.random_state,
                **self.splitter_params,
            )
        elif self.type == "time_series_split":
            
            splitter = TimeSeriesSplit(
                n_splits=self.n_splits,
                **self.splitter_params,
            )
        elif self.type == "group_kfold":
            splitter = GroupKFold(
                n_splits=self.n_splits,
                **self.splitter_params,
            )
        elif self.type == "leave_one_group_out":
            splitter = LeaveOneGroupOut()
            if "groups" not in self.splitter_params:
                raise ValueError("groups must be specified in splitter_params for leave_one_group_out")
        elif self.type == "leave_p_groups_out":
            if 'n_groups' not in self.splitter_params:
                raise ValueError("n_groups must be specified in splitter_params for leave_p_groups_out")
            splitter = LeavePGroupsOut(
                n_groups=self.splitter_params['n_groups']
            )
        else:
            raise ValueError(f"Unsupported splitter type: {self.type}")
        return splitter
    
    def _sample(self):
        """
        Samples training and testing indices from the loaded dataset, optionally using stratification.
        Splits the data into training and testing sets, records the sampling time, and stores the resulting indices.

        Raises
        ------
        ValueError
            If data is not loaded, or if the specified stratify column is not found, or if ``stratify`` is invalid.

        Side Effects
        ------------
        Sets ``self._train_indices``, ``self._test_indices``, and ``self.data_sample_time``.
        
        Logs the time taken for sampling.
        """
        if not hasattr(self, "_X"):
            raise ValueError("Data not loaded. Cannot sample.")
        if not hasattr(self, "_y"):
            raise ValueError("Data not loaded. Cannot sample.")
        if self._X is None or self._y is None:
            raise ValueError("Data not loaded. Cannot sample.")
        start_time = time.process_time()
        # Use stratification, if specified
        if self.stratified is True:
            y = self._y
        elif self.stratified is False:
            y = None
        else:
            assert self.stratified in self._X.columns, f"Stratify column {self.stratified} not found in X columns"
            y = self._X[self.stratified]
        splitter = self._initialize_splitter(
            y=y
        )
        indices = range(len(self._X))
        if self.type in ["leave_one_group_out", "leave_p_groups_out"]:
            if "groups" not in self.splitter_params:
                # Set groups to self._y
                self.splitter_params["groups"] = self._y
            splits = list(splitter.split(indices, **self.splitter_params))
        else:
            splits = list(splitter.split(indices) if self.stratified else splitter.split(self._X))
        if self.split != -1:
            self._train_indices, self._test_indices = splits[self.split]
            end_time = time.process_time()
        else:
            # Use all splits for both training and testing
            train_indices = []
            test_indices = []
            for train_idx, test_idx in splits:
                train_indices.extend(train_idx)
                test_indices.extend(test_idx)
            # Create lists of lists of indices
            self._train_indices = train_indices
            self._test_indices = test_indices
            end_time = time.process_time()
        self.data_sample_time = end_time - start_time
        logger.info(f"Data sampled in {self.data_sample_time:.2f} seconds")
        self.X_train = self._X.iloc[self._train_indices].reset_index(drop=True)
        self.y_train = self._y.iloc[self._train_indices].reset_index(drop=True)
        self.X_test = self._X.iloc[self._test_indices].reset_index(drop=True)
        self.y_test = self._y.iloc[self._test_indices].reset_index(drop=True)
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