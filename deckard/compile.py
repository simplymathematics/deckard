from pathlib import Path
import pandas as pd
import json
import yaml
import logging
import paretoset
from omegaconf import OmegaConf
from joblib import Parallel, delayed
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .utils import ConfigBase
from .impute import ResultImputerConfig
logger = logging.getLogger(__name__)

@dataclass
class ResultFolderConfig(ConfigBase):
    directory: str = "outputs/"
    params_regex : str = "**/params.yaml"
    scores_regex : str = "**/scores.json"
    output_file : str = "outputs/combined_results.csv"
    
    def __post_init__(self):
        self.directory = Path(self.directory).as_posix()
        glob = Path(self.directory).glob
        self.params_files = sorted(list(glob(self.params_regex)), key=lambda f: f.parent.as_posix())
        self.score_files = sorted(list(glob(self.scores_regex)), key=lambda f: f.parent.as_posix())
        # Remove any score files that do not have a corresponding params file and vice versa
        score_file_stems = set([f.parent.as_posix() for f in self.score_files])
        params_file_stems = set([f.parent.as_posix() for f in self.params_files])
        common_stems = score_file_stems.intersection(params_file_stems)
        logger.info(f"Found {len(self.score_files)} score files and {len(self.params_files)} params files before matching.")
        self.score_files = [f for f in self.score_files if f.parent.as_posix() in common_stems]
        self.params_files = [f for f in self.params_files if f.parent.as_posix() in common_stems]
    
    def __iter__(self):
        for score_file, param_file in zip(self.score_files, self.params_files):
            yield score_file, param_file
    
    def _read_pair(score_file, params_file):
            with open(score_file, "r") as sf:
                score_dict = json.load(sf)
            score_df = pd.json_normalize(score_dict)
            with open(params_file, "r") as pf:
                params_dict = yaml.safe_load(pf)
            params_df = pd.json_normalize(params_dict)
            combined_df = pd.concat([params_df, score_df], axis=1)
            return combined_df
    
    def __call__(self):
        big_df = pd.DataFrame()
        

        pairs = list(zip(self.score_files, self.params_files))
        if pairs:
            results = Parallel(n_jobs=-1)(delayed(self._read_pair)(s, p) for s, p in pairs)
            # filter out any None results and concatenate
            results = [r for r in results if r is not None]
            if results:
                big_df = pd.concat(results, ignore_index=True)  
        if self.output_file:
            big_df.to_csv(self.output_file, index=False)
        return big_df
    
    @staticmethod
    def from_csv_static(csv_file: str):
        df = pd.read_csv(csv_file)
        return df

@dataclass
class ParetoConfig(ConfigBase):
    metric: str
    direction: str

    def __post_init__(self):
        directions = ["maximize", "mininimize", "diff"]
        assert self.direction in directions, f"Direction must be one of {directions}, got {self.direction}."
        self.direction = self.direction.replace("imize", "")  # standardize to "max" or "min" or "diff"
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.direction == "diff":
            # For "diff", we want to find the point that is farthest from the line connecting the min and max points
            min_point = df.loc[df[self.metric].idxmin()]
            max_point = df.loc[df[self.metric].idxmax()]
            line_vec = max_point - min_point
            line_vec_norm = line_vec / line_vec.norm()
            distances = df.apply(lambda row: ((row - min_point) - ((row - min_point).dot(line_vec_norm)) * line_vec_norm).norm(), axis=1)
            pareto_df = df.loc[distances.idxmax()].to_frame().T
        elif self.direction == "max":
            pareto_df = df.loc[df[self.metric].idxmax()].to_frame().T
        elif self.direction == "min":
            pareto_df = df.loc[df[self.metric].idxmin()].to_frame().T
        else:
            raise ValueError(f"Unknown direction: {self.direction}")
        return pareto_df
        
@dataclass
class ParetoConfigList(ConfigBase):
    pareto_configs: list[ParetoConfig]
    
    def __post_init__(self):
        directions = ["max", "min", "diff"]
        for config in self.pareto_configs:
            assert config.direction in directions, f"Direction must be one of {directions}, got {config.direction}."
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        subset_df = df.copy()
        subset_df = paretoset.paretoset(subset_df, sense=subset_df.apply(lambda row: [config.direction for config in self.pareto_configs], axis=1).tolist())
        return subset_df

@dataclass
class ResultFormatterConfig(ConfigBase):
    metrics: list
    params: list = []
    sig_figs: list 
    replace : dict = {}
    
    def __post_init__(self):
        assert len(self.metric_columns) == len(self.sig_figs), "Length of metric_columns and sig_figs must be the same."
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        formatted_df = df.copy()
        for col, sig_fig in zip(self.metric_columns, self.sig_figs):
            formatted_df[col] = formatted_df[col].round(sig_fig)
        if len(self.params) == 0:
            tmp_list = [col for col in formatted_df.columns if col not in self.metrics]
            # Drop columns that do not change across rows (i.e. are not parameters)
            for col in tmp_list:
                if formatted_df[col].nunique() <= 1:
                    formatted_df = formatted_df.drop(columns=[col])
                else:
                    self.params.append(col)
        
        # Keep metrics and params, drop any other columns
        keep_cols = self.params + self.metrics
        formatted_df = formatted_df[keep_cols]
        
        # Replace column names according to the replace dict
        for old, new in self.replace.items():
            formatted_df.columns = formatted_df.columns.str.replace(old, new)
        return formatted_df

class CompileConfig(ConfigBase):
    csv_file: str 
    config: str 
    
    def __post_init__(self):
        assert Path(self.config).is_file(), f"Config file {self.config} does not exist."
        self.config_dict = OmegaConf.to_container(OmegaConf.load(self.config))
        # Find relevant keys for ResultFolderConfig
        relevant_keys = {k: v for k, v in self.config_dict.items() if k in ResultFolderConfig.__annotations__}
        self.path = ResultFolderConfig(**relevant_keys)
        
        imputer_keys = self.config_dict.get("imputer", {})
        if len(imputer_keys) > 0:
            self.imputer = ResultImputerConfig(**imputer_keys)
        else:
            self.imputer = None
        
        pareto_keys = self.config_dict.get("pareto", {})
        if len(pareto_keys) > 0:
            self.pareto = ParetoConfigList(pareto_configs=[ParetoConfig(**pc) for pc in pareto_keys.get("pareto_configs", [])])
        else:
            self.pareto = None
        formatter_keys = self.config_dict.get("formatter", {})
        if len(formatter_keys) > 0:
            self.formatter = ResultFormatterConfig(**formatter_keys)
        else:
            self.formatter = None
    
    def __call__(self):
        results = self.path()
        if self.imputer is not None:
            results = self.imputer(results)
        if self.formatter is not None:
            results = self.formatter(results)
        if self.pareto is not None:
            results = self.pareto(results)
        Path(self.csv_file).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.csv_file, index=False)
        return results
    
        
    
    
    
    
    