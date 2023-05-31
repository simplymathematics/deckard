import pandas as pd
from pathlib import Path
import json
import logging
from typing import List

logger = logging.getLogger(__name__)


def parse_folder(
    folder,
    exclude=[
        "probabilities",
        "predictions",
        "plots",
        "ground_truth",
        "adv_predictions",
        "adv_probabilities",
        "samples",
    ],
) -> pd.DataFrame:
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


def parse_results(result_dir, flatten=True):
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
    logger.info(f"Parsing {total} folders...")
    for folder in Path(result_dir).iterdir():
        tmp = parse_folder(folder)
        if flatten is True:
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
            if i == len(key_arr) - 1:
                current[key] = val
            else:
                current[key] = {}
        else:
            if type(current[key]) is not dict:
                logger.info(
                    "Given dictionary is not compatible with key structure requested",
                )
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


def find_results(df, kwargs: dict = {}) -> pd.DataFrame:
    """
    Finds the results of a dataframe that matches the given kwargs.
    """
    logger.debug("Finding best results...")
    for col in kwargs.keys():
        logger.info(f"Finding best results for {col} = {kwargs[col]}...")
        logger.info(f"Shape before: {df.shape}")
        df = df[df[col] == kwargs[col]]
        logger.info(f"Shape after: {df.shape}")
    return df


def drop_static_columns(df) -> pd.DataFrame:
    """
    Drop columns that contain only one unique value.
    :param df: Pandas dataframe
    :return: Pandas dataframe with static columns dropped
    """
    logger.debug("Dropping static columns...")
    for col in df.columns:  # Loop through columns
        if (
            isinstance(df[col].iloc[0], list) and len(df[col].iloc[0]) == 1
        ):  # Find columns that contain lists of length 1
            df[col] = df[col].apply(lambda x: x[0])
        try:
            if (
                len(df[col].unique()) == 1
            ):  # Find unique values in column along with their length and if len is == 1 then it contains same values
                df.drop([col], axis=1, inplace=True)
        except:  # noqa E722
            pass
    return df


def save_results(report_folder, results_file, delete_columns=[]) -> str:
    """
    Compile results from a folder of reports and save to a csv file; return the path to the csv file. It will optionally delete columns from the results.
    """
    logger.info("Compiling results...")
    results = parse_results(report_folder)
    for col in delete_columns:
        try:
            results.drop(col, axis=1, inplace=True)
        except KeyError as e:
            logger.warning(e)
            logger.warning(f"Column {col} not found in results. Skipping.")
            pass
    results.to_csv(results_file)
    assert Path(
        results_file,
    ).exists(), f"Results file {results_file} does not exist. Something went wrong."
    logger.debug(f"Results saved to {results_file}")
    return results_file


def find_best_params(filename, scorer, control_for=None):
    """ """
    logger.info("Finding best params...")
    assert Path(filename).exists(), f"Results file {filename} does not exist."
    results = pd.read_csv(filename, index_col=0)
    big_list = results[control_for].unique() if control_for else []
    if len(big_list) <= 1:
        best_df = results
    else:
        best_df = pd.DataFrame()
        for params in big_list:
            results = find_results(results, kwargs={control_for: params})
            sorted = results.sort_values(by=scorer, ascending=False).head(1)
            best_df = pd.concat([best_df, sorted])
            best_df = best_df.reset_index(drop=True)
    indexes = [Path(x).name for x in best_df["files.path"]]
    best_df["files.path"] = indexes
    return best_df


def delete_these_columns(df, columns) -> pd.DataFrame:
    """
    Delete columns from a dataframe.
    :param df: dataframe
    :param columns: list of columns to delete
    :return: dataframe
    """
    for col in columns:
        logger.info(f"Deleting column {col}...")
        del df[col]
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report_folder", type=str, default="reports")
    parser.add_argument("--results_file", type=str, default="results.csv")
    parser.add_argument("--scorer", type=str, default="accuracy")
    parser.add_argument("--scorer_minimize", type=bool, default=False)
    parser.add_argument("--default_param_file", type=str, default="params.yaml")
    parser.add_argument("--output_folder", type=str, default="best_models")
    parser.add_argument("--control_for", type=str, default=None)
    parser.add_argument("--exclude", type=list, default=None, nargs="*")
    parser.add_argument("--verbose", type=str, default="INFO")
    parser.add_argument(
        "--kwargs",
        type=list,
        default=None,
        nargs="*",
    )
    parser.add_argument("--delete_columns", type=list, nargs="+", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=args.verbose)
    report_folder = args.report_folder
    results_file = args.results_file
    scorer = args.scorer
    default_param_file = args.default_param_file
    output_folder = args.output_folder
    control_for = args.control_for
    kwargs = {}
    if args.kwargs is not None and len(args.kwargs > 0):
        for entry in args.kwargs:
            entry = "".join(entry)
            value = entry.split("=")[1]
            if str(value).isnumeric():
                if int(value) == float(value):
                    value = int(value)
                else:
                    value = float(value)
            kwargs[entry.split("=")[0]] = value
    columns_to_delete = args.delete_columns
    report_file = save_results(
        report_folder,
        results_file,
        delete_columns=columns_to_delete,
    )
    assert Path(
        report_file,
    ).exists(), f"Results file {report_file} does not exist. Something went wrong."
