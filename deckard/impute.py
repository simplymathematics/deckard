from typing import Literal
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from pathlib import Path
import pandas as pd
import json
import yaml
import logging


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .utils import ConfigBase
from .compile import CompileConfig

metric_columns = [
    "accuracy",
    "membership_inference_accuracy",
    "evasion_success",
    "inferred_sex_accuracy",
]
sig_figs = [4, 2, 2, 3]
csv_file = "outputs/combined_results.csv"

big_df = CompileConfig().from_csv(csv_file)
# All numeric columns are relevant for imputation
relevant_columns = big_df.select_dtypes(include=["number"]).columns.tolist()


for metric in metric_columns:
    assert metric in relevant_columns, f"Metric column '{metric}' not found in DataFrame or is not numeric."







class ResultImputerConfig(ConfigBase):
    metric_columns: list 
    sig_figs: list 
    imputer_type: Literal["knn", "iterative"] = "knn"
    imputer_params: dict = {}
    
    def __post_init__(self):
        assert len(self.metric_columns) == len(self.sig_figs), "Length of metric_columns and sig_figs must be the same."
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        imputed_df = df.copy().reset_index(drop=True)
        if self.imputer_type == "knn":
            imputer = KNNImputer(**self.imputer_params)
        elif self.imputer_type == "iterative":
            
            imputer = IterativeImputer(**self.imputer_params)
        else:
            raise ValueError(f"Unknown imputer_type: {self.imputer_type}")
        
        relevant_columns = self.metric_columns
        imputed_values = imputer.fit_transform(imputed_df[relevant_columns])
        imputed_df[relevant_columns] = imputed_values
        
        # Round the imputed metrics to the specified significant figures
        for col, sig_fig in zip(self.metric_columns, self.sig_figs):
            imputed_df[col] = imputed_df[col].round(sig_fig)
        
        return imputed_df
    
big_df = ResultImputerConfig(metric_columns=metric_columns, sig_figs=sig_figs)(big_df)