import warnings
import optuna
import logging
import pandas as pd
from typing import Union
from pathlib import Path
import yaml
import argparse
from hydra.core.hydra_config import HydraConfig

from ..utils import save_data, create_parser_from_function

# suppress future warning
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

def parse_study_name(study_name:str, schema:Union[dict, str]) -> pd.DataFrame:
    """Parses a study name using a dictionary, returning a pd.DataFrame with columns dictated by the schema keys and values given by the schema variables"""
    if isinstance(schema, str):
        assert Path(schema).exists(), f"Schema must be a dictionary or a file path."
        with open(schema, "r") as f:
            conf = yaml.safe_load(f)
            schema = conf.pop("schema", conf)
    schema_copy = schema.copy()
    sep = schema_copy.pop("sep", "_")
    name_list = study_name.split(sep)
    meta_df = pd.DataFrame()
    other_keys = ["sep"]
    for k,v in schema_copy.items():
        if k in other_keys:
            continue
        elif isinstance(v, int):
            try:
                meta_df[k] = [name_list[v]]
            except IndexError as e:
                meta_df[k] = None
        elif isinstance(v, str):
            assert len(v.split(":")) == 2, f"Schema value should either be a an integer or a an inclusive range in the form first:last. Got {v}"
            start, end = map(int, v.split(":"))
            end = min(end, len(name_list) - 1)
            meta_df[k] = sep.join(name_list[start:end + 1])
        else:
            raise ValueError("Unknown value type for schema entry:", type(v))
    return meta_df




def clean_column_names(df):
    cols = df.columns
    clean_cols = []
    for col in cols:
        if col.startswith("values_") or col.startswith("params_"):
            col = col[7:]
            clean_cols.append(col)
        elif col.startswith("++") or col.startswith("~~"):
            col = col[2:]
            clean_cols.append(col)
        else:
            clean_cols.append(col)
    df.columns = clean_cols
    return df

def parse_studies(optuna_db:str, schema:Union[str, dict]) -> pd.DataFrame:
    study_names = optuna.study.get_all_study_names(storage=optuna_db)

    df = pd.DataFrame()
    for name in study_names:
        study = optuna.study.load_study(storage=optuna_db, study_name=name)
        tmp_df = study.trials_dataframe()
        meta_df = parse_study_name(study_name=name, schema = schema)
        tmp_df = tmp_df.merge(meta_df, how="cross")
        df = pd.concat([df, tmp_df])
    df = clean_column_names(df)
    return df

compile_results_parser = argparse.ArgumentParser(description="Parse Optuna studies and compile results")
compile_results_parser.add_argument("--optuna-db", type=str, required=True, help="Path to Optuna database")
compile_results_parser.add_argument("--output-file", type=str, required=True, help="Output CSV file path")
compile_results_parser.add_argument("--schema", type=str, required=True, help="Path to schema YAML file")


def compile_results_main( schema:str, output_file:str, optuna_db:Union[str,type(None)]=None):
    # Check if schema is string or dict
    if optuna_db is None:
        hydra_cfg = HydraConfig.get()
        sweeper = hydra_cfg.get("sweeper", {})
        optuna_db = sweeper.get("storage", ValueError(f"optuna_db must be specified or available as a hydra config. Got: {optuna_db}."))
    schema_yaml = yaml.safe_dump(schema)
    if isinstance(schema_yaml, dict):
        pass
    else:
        schema = str(Path(schema).absolute())
        assert Path(schema).is_file(), f"Schema must be a dictionary or a valid file. Got {schema.absolute()}."
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    optuna_db = optuna_db
    output_file = str(output_path)
    
    df = parse_studies(optuna_db=optuna_db, schema = schema)
    save_data(data=df, filepath=output_file)

compile_results_parser = create_parser_from_function(compile_results_main)
if __name__ == "__main__":
    args = compile_results_parser.parse_args()
    compile_results_main(**args)
    
    
    
    
    