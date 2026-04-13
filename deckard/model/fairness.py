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

from ..data.fairness import FairnessDataConfig
from . import ModelConfig

@dataclass
class FairnessModelConfig(ModelConfig):
    """
    A model configuration that extends ModelConfig to support fairness-aware evaluation.
    
    Trains normally on X_train, y_train, but uses group information from X_test, y_test
    for fairness-aware scoring across demographic groups.
    
    Attributes:
    -----------
    fairness_data : FairnessDataConfig or None
        Configuration containing group information and fairness metrics.
    """
    
    data: Union[FairnessDataConfig, None] = None
    
    def _classification_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Compute classification scores with fairness metrics by group.
        
        Returns base scores plus per-group fairness metrics if fairness_data is available.
        """
        scores = super()._classification_scores(y_true, y_pred)
        
        if self.data is not None and hasattr(self.data, 'groups'):
            fairness_scores = self._compute_group_fairness_scores(y_true, y_pred)
            scores.update(fairness_scores)
        
        return scores
    
    def _regression_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Compute regression scores with fairness metrics by group.
        
        Returns base scores plus per-group fairness metrics if fairness_data is available.
        """
        scores = super()._regression_scores(y_true, y_pred)
        
        if self.data is not None and hasattr(self.data, 'groups'):
            fairness_scores = self._compute_group_fairness_scores(y_true, y_pred)
            scores.update(fairness_scores)
        
        return scores
    
    def _compute_group_fairness_scores(self, y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Compute fairness metrics stratified by protected groups.
        
        Returns per-group performance metrics to assess disparate impact.
        """
        fairness_scores = {}
        
        if not hasattr(self.data, 'groups') or self.data.groups is None:
            return fairness_scores
        
        groups = self.data.groups
        
        for group_name in groups.unique():
            mask = groups == group_name
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            if len(group_y_true) > 0:
                if self.classifier:
                    group_scores = super()._classification_scores(group_y_true, group_y_pred)
                else:
                    group_scores = super()._regression_scores(group_y_true, group_y_pred)
                
                for metric, value in group_scores.items():
                    fairness_scores[f"{group_name}_{metric}"] = value
        
        return fairness_scores