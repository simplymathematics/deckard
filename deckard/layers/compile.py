
import pandas as pd
from pathlib import Path
import json
import yaml
import logging 
from sklearn.model_selection import ParameterGrid
from mergedeep import merge
from deckard.base.hashable import my_hash

logger = logging.getLogger(__name__)

def parse_folder(folder, exclude = ['probabilities', 'predictions', 'plots', 'ground_truth']) -> pd.DataFrame:
    """
    
    
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
    new_results = pd.DataFrame()
    logger.debug("Flattening results...")
    for col in results.columns:
        tmp = pd.json_normalize(results[col])
        new_results = pd.concat([new_results, tmp], axis=1)
    return new_results


def parse_results(result_dir, delete_columns = []):
    result_dir = Path(result_dir)
    assert result_dir.is_dir(), f"Result directory {result_dir} does not exist."
    results = pd.DataFrame()
    logger.debug("Parsing results...")
    for folder in result_dir.glob("*/*"):
        tmp = parse_folder(folder)
        tmp = flatten_results(tmp)
        results = pd.concat([results, tmp])
    # results = drop_static_columns(results, delete_columns =delete_columns)
    return results



def set_for_keys(my_dict, key_arr, val):
    """
    Set val at path in my_dict defined by the string (or serializable object) array key_arr
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

def unflatten_results(df, sep="."):
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

def find_best_subset(df, kwargs: dict = {}):
    logger.debug("Finding best subset...")
    for col in kwargs.keys():
        df = df[df[col] == kwargs[col]]
    select_these = df
    return select_these

def create_param_files_from_df(df, default_param_file = "queue/default.yaml", output_dir = "best" ):
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

def drop_static_columns(df, delete_columns = []):
    logger.debug("Dropping static columns...")
    for col in df.columns:  # Loop through columns
        if isinstance(df[col].iloc[0], list) and len(df[col].iloc[0]) == 1:  # Find columns that contain lists of length 1
            df[col] = df[col].apply(lambda x: x[0])
        if col in delete_columns:
            df.drop([col], axis=1, inplace=True)
        try:
            if len(df[col].unique()) == 1:  # Find unique values in column along with their length and if len is == 1 then it contains same values
                df.drop([col], axis=1, inplace=True) 
        except:
            pass
    return df


def compile_results(report_folder, results_file, delete_columns = []):
    logger.info("Compiling results...")
    results = parse_results(report_folder, delete_columns = delete_columns)
    results.to_csv(results_file)
    assert Path(results_file).exists(), f"Results file {results_file} does not exist. Something went wrong."
    logger.debug(f"Results saved to {results_file}")
    return results_file
    
def find_best_params(filename, scorer, default_param_file, output_folder, control_for):
    logger.info("Finding best params...")
    assert Path(filename).exists(), f"Results file {filename} does not exist."
    results = pd.read_csv(filename, index_col = 0)
    big_list = results[control_for].unique()
    if len(big_list) <= 1:
        best_df = results
    else:
        best_df = pd.DataFrame()
        for params in big_list:
            subset = find_best_subset(results, kwargs = {control_for : params})
            sorted = subset.sort_values(by =scorer, ascending = False).head(1)
            best_df = pd.concat([best_df, sorted])
            best_df = best_df.reset_index(drop = True)
    indexes = [Path(x).name for x in best_df['files.path']]
    best_df['files.path'] = indexes
    return best_df



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
    report_file = compile_results(report_folder, results_file, delete_columns = ["Unnamed: 0", "model.init.random_state"])
    assert Path(report_file).exists(), f"Results file {report_folder} does not exist. Something went wrong."
    best_df = find_best_params(report_file, scorer, default_param_file, output_folder, control_for)
    param_files = create_param_files_from_df(best_df, default_param_file = default_param_file, output_dir = output_folder)