from typing import Optional, Union
import pandas as pd
from dataclasses import field, dataclass

from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
    chi2,
    f_classif,
    f_regression,
    r_regression,
)

from .data import DataConfig

@dataclass
class FairnessDataConfig(DataConfig):
    """
    Extended DataConfig class that overloads key methods to operate on pandas groupby objects.
    
    This allows stratified analysis of fairness metrics across different demographic groups.
    """
    
    groupby_columns: Optional[Union[str,list]] = None
    groups_ :pd.api.typing.DataFrameGroupBy = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize with groupby_column support."""
        super().__post_init__()
        if self.groupby_columns is None:
            raise ValueError("groupby_column must be specified for FairnessDataConfig")
    
    
    def _load_data(self):
        super()._load_data()
        assert hasattr(self, "_X"), RuntimeError("self.X_ not found while loading FairnessDataConfig")
        assert hasattr(self, "_y"), RuntimeError("self.y_ not found whilte loading FairnessDataConfig")
        assert isinstance(self._X, pd.DataFrame), ValueError("Expected a dataframe for self.X_")
        for col in self.groupby_columns:
            assert col in self._X.columns
        self.groups_ = self._X.groupby(by=self.groupby_columns)
        return self
        
    def _sample(self):
        """Override _sample to handle groupby objects for fairness analysis.
        
        Keeps X_train, y_train the same for all groups, but creates separate
        X_test, y_test groups based on the groupby columns.
        """
        # Call parent _sample to get standard train/test split
        super()._sample()
        
        # Store the original X_test and y_test
        self.X_test_groups = {}
        self.y_test_groups = {}
        
        # Create grouped versions of test data
        X_test_grouped = self.X_test.groupby(by=self.groupby_columns)
        
        for group_name, group_data in X_test_grouped:
            group_indices = group_data.index
            self.X_test_groups[group_name] = group_data
            self.y_test_groups[group_name] = self.y_test.loc[group_indices]
        
        
    def _score(self) -> dict:
        """Compute fairness scores for each group."""
        scores = {}
        for group_name, group_data in self.groups_:
            group_indices = group_data.index
            y_group = self._y.loc[group_indices]
            
            if self.classifier:
                scores[group_name] = self._classification_feature_scores_for_group(y_group)
            else:
                scores[group_name] = self._regression_feature_scores_for_group(y_group)
        
        return scores

    def _compute_class_counts(self, y: pd.Series) -> dict:
        """Compute class counts for each group."""
        counts = {}
        for group_name, group_data in self.groups_:
            group_indices = group_data.index
            y_group = self._y.loc[group_indices]
            counts[group_name] = y_group.value_counts().to_dict()
        
        return counts

    def _classification_feature_scores_for_group(self, y_group: pd.Series) -> dict:
        """Compute classification feature scores for a specific group."""
        group_indices = y_group.index
        X_group = self._X.loc[group_indices]
        
        scores = {
            "f_classif": f_classif(X_group, y_group)[0],
            "mutual_info": mutual_info_classif(X_group, y_group, random_state=self.random_state),
            "chi2": chi2(X_group, y_group)[0],
        }
        return scores

    def _regression_feature_scores_for_group(self, y_group: pd.Series) -> dict:
        """Compute regression feature scores for a specific group."""
        group_indices = y_group.index
        X_group = self._X.loc[group_indices]
        
        scores = {
            "f_regression": f_regression(X_group, y_group)[0],
            "mutual_info": mutual_info_regression(X_group, y_group, random_state=self.random_state),
            "r_regression": r_regression(X_group, y_group),
        }
        return scores


       
       
            
        
    
    