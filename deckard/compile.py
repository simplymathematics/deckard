from pathlib import Path
import pandas as pd
import json
import yaml
import logging
import paretoset
from tqdm import tqdm
from inspect import signature
from omegaconf import ListConfig, OmegaConf, DictConfig
from joblib import Parallel, delayed
from dataclasses import dataclass, field


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
    result_output_file : str = "outputs/combined_results.csv"
    
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
    
    def _read_pair(self, score_file, params_file):
        with open(score_file, "r") as sf:
            score_dict = json.load(sf)
        score_df = pd.json_normalize(score_dict)
        with open(params_file, "r") as pf:
            params_dict = yaml.safe_load(pf)
        params_df = pd.json_normalize(params_dict)
        combined_df = pd.concat([params_df, score_df], axis=1)
        score_file_id = score_file.parent.stem
        params_file_id = params_file.parent.stem
        assert score_file_id == params_file_id, f"Score file {score_file.parent.stem} and params file {params_file.parent.stem} do not match."
        combined_df['files.score_file'] = score_file.as_posix()
        combined_df['files.params_file'] = params_file.as_posix()
        combined_df['experiment_name'] = combined_df['experiment_name'].apply(lambda x: x.replace("*", score_file_id))
        return combined_df
    
    def __call__(self):
        if self.result_output_file and Path(self.result_output_file).is_file():
            big_df = pd.read_csv(self.result_output_file)
        else:
            big_df = pd.DataFrame()
        pairs = list(zip(self.score_files, self.params_files))
        if pairs:
            results = Parallel(n_jobs=-1)(delayed(self._read_pair)(s, p) for s, p in tqdm(pairs, desc=f"Reading {len(pairs)} results from {self.directory}"))
            # filter out any None results and concatenate
            results = [r for r in results if r is not None]
            if results:
                big_df = pd.concat(results, ignore_index=True)             
        if self.result_output_file:
            self.save_scores(scores = big_df, filepath=self.result_output_file)
        return big_df
    

@dataclass
class ParetoConfig(ConfigBase):
    metric: str
    direction: str

    def __post_init__(self):
        directions = ["maximize", "minimize", "diff"]
        assert self.direction in directions, f"Direction must be one of {directions}, got {self.direction}."
        self.direction = self.direction.replace("imize", "")  # standardize to "max" or "min" or "diff"
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        subset_df = df.copy()
        pareto_df = paretoset.paretoset(subset_df[[self.metric]], sense=[self.direction])
        subset_df = subset_df.loc[pareto_df].reset_index(drop=True)
        return subset_df
        
@dataclass
class ParetoConfigList(ConfigBase):
    pareto_configs: list[ParetoConfig]
    
    def __post_init__(self):
        directions = ["max", "min", "diff"]
        for config in self.pareto_configs:
            assert config.direction in directions, f"Direction must be one of {directions}, got {config.direction}."
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        subset_df = df.copy()
        senses = []
        metrics = []
        for config in self.pareto_configs:
            if config.direction in ["max", "min", "diff"]:
                senses.append(config.direction)
            else: # validate direction is acceptable
                raise ValueError(f"Unknown direction: {config.direction}")
            if config.metric not in subset_df.columns: # validate that metric exists in dataframe
                raise ValueError(f"Metric {config.metric} not found in dataframe columns.")
            else:
                metrics.append(config.metric)
        pareto_df = paretoset.paretoset(subset_df[metrics], sense=senses)
        # Get the original rows corresponding to the pareto optimal points
        subset_df = subset_df.loc[pareto_df].reset_index(drop=True)
        return subset_df

@dataclass
class ResultFormatterConfig(ConfigBase):
    sig_figs: ListConfig
    metrics : ListConfig
    params: ListConfig = field(default_factory=list)
    replace : DictConfig = field(default_factory=dict)
    
    def __post_init__(self):
        assert len(self.metrics) == len(self.sig_figs), "Length of metric_columns and sig_figs must be the same."
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        formatted_df = df.copy()
        for col, sig_fig in zip(self.metrics, self.sig_figs):
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
    result_output_file: str 
    compile_config_file: str = None 
    
    def __post_init__(self):
        if self.compile_config_file is not None:
            assert Path(self.compile_config_file).is_file(), f"Config file {self.compile_config_file} does not exist."
            self.config_dict = OmegaConf.to_container(OmegaConf.load(self.compile_config_file))
        else:
            self.config_dict = self.__dict__
        # Use signature to get relevant keys for ResultFolderConfig
        relevant_keys = {}
        folder_config_params = signature(ResultFolderConfig).parameters
        for key in folder_config_params:
            if key in self.config_dict:
                relevant_keys[key] = self.config_dict[key]
        self.path = ResultFolderConfig(**relevant_keys)
        
        imputer_keys = self.config_dict.get("impute", {})
        if len(imputer_keys) > 0:
            self.imputer = ResultImputerConfig(**imputer_keys)
        else:
            self.imputer = None
        pareto = self.config_dict.get("pareto", {})
        if len(pareto) > 0:
            self.pareto = ParetoConfigList(pareto_configs=[ParetoConfig(**pc) for pc in pareto])
        else:
            self.pareto = None
        formatter_keys = self.config_dict.get("format", {})
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
        Path(self.result_output_file).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.result_output_file, index=False)
        return results
    
        
    
    
    
    
    