import pandas as pd
from pathlib import Path
import json
import logging
from tqdm import tqdm
import yaml
import argparse


logger = logging.getLogger(__name__)


def flatten_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): a dataframe with dictionaries as entries in some columns

    Returns:
        pd.DataFrame: a dataframe with the dictionaries flattened into columns using pd.json_normalize
    """
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    for col in tqdm(df.columns, desc="Flattening columns"):
        if isinstance(df[col][0], dict):
            tmp = pd.json_normalize(df[col].fillna({i: {} for i in df[col].index}))
            tmp.columns = [f"{col}.{subcol}" for subcol in tmp.columns]
            tmp.index = df.index
            df = pd.merge(df, tmp, left_index=True, how="outer", right_index=True)
            if f"files.{col}_file" in tmp:
                df[col] = Path(tmp[f"files.{col}_file"]).apply(lambda x: x.stem)
            else:
                df[col] = tmp.index
        else:
            df = pd.concat([df, df[col]], axis=1)
    return df


def parse_folder(
    folder,
    files=["params.yaml", "score_dict.json"],
    other_files=False,
) -> pd.DataFrame:
    """
    Parse a folder containing files and return a dataframe with the results, excluding the files in the exclude list.
    :param folder: Path to folder containing files
    :param files: List of files to parse. Defaults to ["params.yaml", "score_dict.json"]. Other files will be added as columns with hrefs.
    :return: Pandas dataframe with the results
    """
    folder = Path(folder)

    logger.debug(f"Parsing folder {folder}...")
    path_gen = []
    for file in files:
        path_gen.extend(folder.glob(f"**/{file}"))
    path_gen.sort()
    path_gen = list(set(path_gen))
    path_gen.sort()
    path_gen = list(set(path_gen))
    path_gen.sort()
    folder_gen = map(lambda x: x.parent, path_gen)
    folder_gen = set(folder_gen)
    results = {}
    for file in tqdm(path_gen, desc="Parsing Specified files"):
        results = read_file(file, results)
    if other_files is True:
        for folder in tqdm(folder_gen, desc="Adding other files to results"):
            results = add_file(folder, path_gen, results)
    df = pd.DataFrame(results).T
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()
    return df


def add_file(folder, path_gen, results):
    all_files = Path(folder).glob("**/*")
    for file in all_files:
        if file not in path_gen:
            if file.parent.name not in results:
                results[file.parent.name] = {}
            results[file.parent.name][file.stem] = file
    return results


def read_file(file, results):
    suffix = file.suffix
    folder = file.parent.name
    stage = file.parent.parent.name
    if folder not in results:
        results[folder] = {}
    if suffix == ".json":
        try:
            retries = locals().get("retries", 0)
            with open(file, "r") as f:
                dict_ = json.load(f)
        except json.decoder.JSONDecodeError as e:
            logger.error(f"Error reading {file}")
            print(f"Error reading {file}. Please fix the file and press Enter.")
            input(
                "Press Enter to continue. The next failure on this file will raise an error.",
            )
            if retries > 1:
                raise e
            else:
                with open(file, "r") as f:
                    dict_ = json.load(f)
                retries += 1
    elif suffix == ".yaml":
        with open(file, "r") as f:
            try:
                dict_ = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error reading {file}")
                print(f"Error reading {file}")
                input("Press Enter to raise the error.")
                raise e
    else:
        raise ValueError(f"File type {suffix} not supported.")
    results[folder]["stage"] = stage
    results[folder].update(dict_)
    return results


def parse_results(folder, files=["score_dict.json", "params.yaml"]):
    df = parse_folder(folder, files=files)
    df = flatten_results(df)
    return df


def save_results(results, results_file, results_folder) -> str:
    """
    Compile results from a folder of reports and save to a csv file; return the path to the csv file. It will optionally delete columns from the results.
    """
    assert isinstance(
        results,
        pd.DataFrame,
    ), f"Results must be a pandas DataFrame, not {type(results)}."
    results_file = Path(results_folder, results_file)
    logger.info(f"Saving data to {results_file}")
    Path(results_file).parent.mkdir(exist_ok=True, parents=True)
    suffix = results_file.suffix
    if suffix == ".csv":
        results.to_csv(results_file, index=True)
    elif suffix == ".xlsx":
        results.to_excel(results_file, index=True)
    elif suffix == ".html":
        results.to_html(results_file, index=True)
    elif suffix == ".json":
        results.to_json(results_file, index=True, orient="records")
    elif suffix == ".tex":
        pretty_model = results_file.stem.replace("_", " ").title()
        results.to_latex(
            results_file,
            index=True,
            escape=True,
            label=f"tab:{results_file.stem}",
            caption=f"{pretty_model} Results",
            header=True,
            position="htbp",
        )
    else:
        raise ValueError(f"File type {suffix} not supported.")
    assert Path(
        results_file,
    ).exists(), f"Results file {results_file} does not exist. Something went wrong."
    return results_file


def load_results(results_file, results_folder) -> pd.DataFrame:
    """
    Load results from a csv file; return the path to the csv file. It will optionally delete columns from the results.
    """
    results_file = Path(results_folder, results_file)
    logger.info(f"Loading data from {results_file}")
    Path(results_folder).mkdir(exist_ok=True, parents=True)
    suffix = results_file.suffix
    if suffix == ".csv":
        results = pd.read_csv(results_file)
    elif suffix == ".xlsx":
        results = pd.read_excel(results_file)
    elif suffix == ".html":
        results = pd.read_html(results_file)
    elif suffix == ".json":
        results = pd.read_json(results_file)
    elif suffix == ".tex":
        pd.read_csv(
            results_file,
            sep="&",
            header=None,
            skiprows=4,
            skipfooter=3,
            engine="python",
        )
    else:
        raise ValueError(f"File type {suffix} not supported.")
    assert Path(
        results_file,
    ).exists(), f"Results file {results_file} does not exist. Something went wrong."
    return results


def compile_main(parse_results, save_results, args):
    logging.basicConfig(level=args.verbose)
    report_folder = args.report_folder
    results_file = args.results_file
    results_folder = args.results_folder
    results = parse_results(report_folder)
    report_file = save_results(results, results_file, results_folder)
    assert Path(
        report_file,
    ).exists(), f"Results file {report_file} does not exist. Something went wrong."


compile_parser = argparse.ArgumentParser()
compile_parser.add_argument("--results_file", type=str, default="results.csv")
compile_parser.add_argument(
    "--report_folder", type=str, default="reports", required=True
)
compile_parser.add_argument("--results_folder", type=str, default=".")
compile_parser.add_argument("--exclude", type=list, default=None, nargs="*")
compile_parser.add_argument("--verbose", type=str, default="INFO")

if __name__ == "__main__":
    args = compile_parser.parse_args()
    compile_main(parse_results, save_results, args)
