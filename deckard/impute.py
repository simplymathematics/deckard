from typing import Literal
from sklearn.experimental import enable_iterative_imputer # N
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
import pandas as pd
from omegaconf import ListConfig
from dataclasses import field, dataclass

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .utils import ConfigBase






@dataclass
class ResultImputerConfig(ConfigBase):
    metric_columns: ListConfig
    sig_figs: ListConfig
    imputer_type: Literal["knn", "iterative", "simple"] = "knn"
    imputer_params: dict = field(default_factory=dict)
    ignore: list = field(default_factory=lambda: ["trial_id", "experiment_id", "config_hash"])
    
    def __post_init__(self):
        assert len(self.metric_columns) == len(self.sig_figs), "Length of metric_columns and sig_figs must be the same."
        
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_df = df.copy().reset_index(drop=True)
        
        if self.imputer_type == "knn":
            imputer = KNNImputer(**self.imputer_params)
        elif self.imputer_type == "simple":
            imputer = SimpleImputer(**self.imputer_params)
        elif self.imputer_type == "iterative":
            
            imputer = IterativeImputer(**self.imputer_params)
            
        else:
            raise ValueError(f"Unknown imputer_type: {self.imputer_type}")
        numeric_columns = imputed_df.select_dtypes(include=['number']).columns.tolist()
        relevant_columns = [col for col in numeric_columns if col not in self.ignore]
        imputer.fit(imputed_df[relevant_columns])
        imputed_values = imputer.transform(imputed_df[relevant_columns])
        imputed_df[relevant_columns] = imputed_values
        # Round the imputed metrics to the specified significant figures
        for col, sig_fig in zip(self.metric_columns, self.sig_figs):
            imputed_df[col] = imputed_df[col].round(sig_fig)
        
        return imputed_df
    
