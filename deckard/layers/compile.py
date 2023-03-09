
import pandas as pd
from pathlib import Path
import json
import yaml
import logging 
from typing import List
from mergedeep import merge
from deckard.base.hashable import my_hash
from tqdm import tqdm
import os


logger = logging.getLogger(__name__)

def parse_folder(folder, exclude = ['probabilities', 'predictions', 'plots', 'ground_truth', "attack_predictions", "attack_probabilities", "samples"]) -> pd.DataFrame:
    """
    Parse a folder containing json files and return a dataframe with the results, excluding the files in the exclude list.
    :param folder: Path to folder containing json files
    :param exclude: List of files to exclude. Default: ['probabilities', 'predictions', 'plots', 'ground_truth'].
    :return: Pandas dataframe with the results
    """
    folder = Path(folder)
    results = {}
    results[folder] = {}
    logger.debug(f"Parsing folder {folder}...")
    for file in folder.glob("*.json"):
        if Path(file).stem in exclude:
            continue
        else:
            with open(file, "r") as f:
                results[folder][Path(file).stem] = json.load(f)
    return pd.DataFrame(results).T

def flatten_results(results):
    """
    Flatten a dataframe containing json files. So that each json dict entry becomes a column with dot notation (e.g. "key1.subkey1")
    :param results: Pandas dataframe containing json files
    """
    new_results = pd.DataFrame()
    logger.debug("Flattening results...")
    for col in results.columns:
        tmp = pd.json_normalize(results[col])
        new_results = pd.concat([new_results, tmp], axis=1)
    return new_results


def parse_results(result_dir,  flatten = True):
    """
    Recursively parse a directory containing json files and return a dataframe with the results.
    :param result_dir: Path to directory containing json files
    :param regex: Regex to match folders to parse. Default: "*/*"
    :param flatten: Whether to flatten the results. Default: True
    :return: Pandas dataframe with the results
    """
    result_dir = Path(result_dir)
    assert result_dir.is_dir(), f"Result directory {result_dir} does not exist."
    results = pd.DataFrame()
    logger.debug("Parsing results...")
    total = len(list(Path(result_dir).iterdir()))
    print(f"Parsing {total} folders...")
    for folder in tqdm(Path(result_dir).iterdir()):
        tmp = parse_folder(folder)
        if flatten == True:
            tmp = flatten_results(tmp)
        tmp = tmp.loc[:, ~tmp.columns.duplicated()]
        results = pd.concat([results, tmp])
    return results



def set_for_keys(my_dict, key_arr, val) -> dict:
    """
    Set val at path in my_dict defined by the string (or serializable object) array key_arr.
    :param my_dict: Dictionary to set value in
    :param key_arr: Array of keys to set value at
    :param val: Value to set
    :return: Dictionary with value set
    """
    current = my_dict
    for i in range(len(key_arr)):
        key = key_arr[i]
        if key not in current:
            if i==len(key_arr)-1:
                current[key] = val
            else:
                current[key] = {}
        else:
            if type(current[key]) is not dict:
                print("Given dictionary is not compatible with key structure requested")
                raise ValueError("Dictionary key already occupied")
        current = current[key]
    return my_dict

def unflatten_results(df, sep=".") -> List[dict]:
    """
    Unflatten a dataframe with dot notation columns (e.g. "key1.subkey1") into a list of dictionaries.
    :param df: Pandas dataframe with dot notation columns
    :param sep: Separator to use. Default: "."
    :return: List of dictionaries
    """
    logger.debug("Unflattening results...")
    result = []
    for _, row in df.iterrows():
        parsed_row = {}
        for idx, val in row.iteritems():
            if val == val:
                keys = idx.split(sep)
                parsed_row = set_for_keys(parsed_row, keys, val)
        result.append(parsed_row)
    return result

def find_subset(df, kwargs: dict = {}) -> pd.DataFrame:
    """
    Finds the subset of a dataframe that matches the given kwargs.
    """
    logger.debug("Finding best subset...")
    for col in kwargs.keys():
        df = df[df[col] == kwargs[col]]
    return df

def create_param_files_from_df(df, default_param_file = "queue/default.yaml", output_dir = "best" ) -> List[Path]:
    logger.debug("Creating param files from dataframe...")
    paths = []
    select_these = df.copy()
    for col in select_these.columns:
        if "data." in str(col)or "model." in str(col) or "files." in str(col)in str(col):
            pass
        else:
            select_these.drop(col, axis = 1, inplace = True)
    with open(default_param_file, 'r') as stream:
        default_params = yaml.safe_load(stream)
    default_params = pd.json_normalize(default_params).iloc[0].to_dict()
    for _, row in select_these.iterrows():
        new = row.copy()
        filename = my_hash(new.to_dict())
        merged = merge(default_params, new.to_dict())
        pathname = my_hash(merged)
        merged['files.path'] = pathname
        un_json = unflatten_results(pd.DataFrame([merged]))[0]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir, filename + ".yaml"), 'w') as outfile:
            yaml.dump(un_json, outfile, default_flow_style=False)
        assert Path(output_dir, filename + ".yaml").exists(), f"Param file {filename} does not exist. Something went wrong."
        logger.debug(f"Param file saved to {output_dir}/{filename}.yaml")
        paths.append(str(Path(output_dir, filename + ".yaml")))
    return paths

def drop_static_columns(df) -> pd.DataFrame:
    """
    Drop columns that contain only one unique value.
    :param df: Pandas dataframe
    :return: Pandas dataframe with static columns dropped
    """
    logger.debug("Dropping static columns...")
    for col in df.columns:  # Loop through columns
        if isinstance(df[col].iloc[0], list) and len(df[col].iloc[0]) == 1:  # Find columns that contain lists of length 1
            df[col] = df[col].apply(lambda x: x[0])
        try:
            if len(df[col].unique()) == 1:  # Find unique values in column along with their length and if len is == 1 then it contains same values
                df.drop([col], axis=1, inplace=True) 
        except:
            pass
    return df


def save_results(report_folder, results_file, delete_columns = []) -> str:
    """
    Compile results from a folder of reports and save to a csv file; return the path to the csv file. It will optionally delete columns from the results.
    """
    logger.info("Compiling results...")
    results = parse_results(report_folder, delete_columns = delete_columns)
    results.to_csv(results_file)
    assert Path(results_file).exists(), f"Results file {results_file} does not exist. Something went wrong."
    logger.debug(f"Results saved to {results_file}")
    return results_file
    
def find_best_params(filename, scorer, control_for = None):
    """
    
    """
    logger.info("Finding best params...")
    assert Path(filename).exists(), f"Results file {filename} does not exist."
    results = pd.read_csv(filename, index_col = 0)
    big_list = results[control_for].unique() if control_for else []
    if len(big_list) <= 1:
        best_df = results
    else:
        best_df = pd.DataFrame()
        for params in big_list:
            subset = find_subset(results, kwargs = {control_for : params})
            sorted = subset.sort_values(by =scorer, ascending = False).head(1)
            best_df = pd.concat([best_df, sorted])
            best_df = best_df.reset_index(drop = True)
    indexes = [Path(x).name for x in best_df['files.path']]
    best_df['files.path'] = indexes
    return best_df

def delete_these_columns(df, columns) -> pd.DataFrame:
    """
    Delete columns from a dataframe.
    :param df: dataframe
    :param columns: list of columns to delete
    :return: dataframe
    """
    for col in columns:
        del df[col]
    return df


def dump_best_model_to_yaml(df, col = None, path = "best_models", scorer = "accuracy", n=1) -> List[str]:
    """
    Dumps the best model to a yaml configuration file.
    :param df: dataframe
    :param col: column to group by. Will find the best configuration for each unique value in the column.
    :param path: path to save the yaml file. Defaults to "best_models".
    :param scorer: column to sort by. Defaults to "accuracy".
    :param n: number of best models to save. Defaults to 1.
    :return: list of yaml files
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    files = []
    if col is None:
        result = df.sort_values(by = scorer, ascending = False).head(n)
    else:
        cols = df[col].unique()
        for col_i in cols:
            result = df[df[col] == col_i].sort_values(by = scorer, ascending = False).head(n)
            del result[scorer]
            with open(f"{path}/{col}.yaml", "w") as f:
                yaml.dump(unflatten_results(result)[0], f)
            files.append(f"{path}/{col}.yaml")
    return files

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_folder", type=str, default="reports")
    parser.add_argument("--results_file", type=str, default="newest_results.csv")
    parser.add_argument("--scorer", type=str, default="accuracy")
    parser.add_argument("--default_param_file", type=str, default="params.yaml")
    parser.add_argument("--output_folder", type=str, default="best")
    parser.add_argument("--control_for", type=str, default="model.init.kernel")
    parser.add_argument("--verbose", type=str, default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.verbose)
    report_folder = args.report_folder
    results_file = args.results_file
    scorer = args.scorer
    default_param_file = args.default_param_file
    output_folder = args.output_folder
    control_for = args.control_for
    report_file = save_results(report_folder, results_file, delete_columns = ["Unnamed: 0", "model.init.random_state"])
    assert Path(report_file).exists(), f"Results file {report_folder} does not exist. Something went wrong."
    results = pd.read_csv(report_file, index_col = 0)
    kwargs = { "data.sample.train_size" : 1000, "data.generate.n_features" : 100}
    tmp = find_subset(results, kwargs = kwargs)
    tmp = delete_these_columns(tmp, ['f1', 'precision', 'recall', 'predict_time', 'fit_time', 'proba_time'])
    best_files = dump_best_model_to_yaml(tmp, col = 'model.init.kernel', path = output_folder, scorer = scorer, n=1)
    for file in best_files:
        assert Path(file).exists(), f"{file} does not exist."