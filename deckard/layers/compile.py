import pandas as pd
from pathlib import Path
import json
import logging
from tqdm import tqdm
import yaml
import numpy as np
from .utils import deckard_nones as nones

logger = logging.getLogger(__name__)


def flatten_results(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if isinstance(df[col][0], dict):
            tmp = pd.json_normalize(df[col])
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


def parse_folder(folder, files=["params.yaml", "score_dict.json"]) -> pd.DataFrame:
    """
    Parse a folder containing files and return a dataframe with the results, excluding the files in the exclude list.
    :param folder: Path to folder containing files
    :param files: List of files to parse. Defaults to ["params.yaml", "score_dict.json"]. Other files will be added as columns with hrefs.
    :return: Pandas dataframe with the results
    """
    folder = Path(folder)
    results = {}
    logger.debug(f"Parsing folder {folder}...")
    path_gen = []
    for file in files:
        path_gen.extend(folder.glob(f"**/{file}"))
    path_gen.sort()
    indices = []
    for file in tqdm(path_gen):
        indices.append(file.parent.name)
        suffix = file.suffix
        folder = file.parent.name
        stage = file.parent.parent.name
        if folder not in results:
            results[folder] = {}
        if suffix == ".json":
            with open(file, "r") as f:
                dict_ = json.load(f)
        elif suffix == ".yaml":
            with open(file, "r") as f:
                dict_ = yaml.safe_load(f)
        else:
            raise ValueError(f"File type {suffix} not supported.")
        results[folder]["stage"] = stage
        results[folder].update(dict_)
    all_files = Path(folder).glob("**/*")
    for file in all_files:
        if file not in path_gen:
            if file.parent.name not in results:
                results[file.parent.name] = {}
            results[file.parent.name][file.stem] = file
    df = pd.DataFrame(results).T
    return df


def merge_defences(results: pd.DataFrame):
    defences = []
    def_gens = []
    for _, entry in results.iterrows():
        defence = []
        if (
            "model.art.pipeline.preprocessor.name" in entry
            and entry["model.art.pipeline.preprocessor.name"] not in nones
        ):
            defence.append(entry["model.art.pipeline.preprocessor.name"])
        if (
            "model.art.pipeline.postprocessor.name" in entry
            and entry["model.art.pipeline.postprocessor.name"] not in nones
        ):
            defence.append(entry["model.art.pipeline.postprocessor.name"])
        if (
            "model.art.pipeline.transformer.name" in entry
            and entry["model.art.pipeline.transformer.name"] not in nones
        ):
            defence.append(entry["model.art.pipeline.transformer.name"])
        if (
            "model.art.pipeline.trainer.name" in entry
            and entry["model.art.pipeline.trainer.name"] not in nones
        ):
            defence.append(entry["model.art.pipeline.trainer.name"])
        ############################################################################################################
        if len(defence) > 1:
            def_gen = [str(x).split(".")[-1] for x in defence]
            defence = "_".join(defence)
        elif len(defence) == 1:
            def_gen = [str(x).split(".")[-1] for x in defence][0]
            defence = defence[0]
        else:
            def_gen = "Control"
            defence = "Control"
        ############################################################################################################
        if defence != []:
            defences.append(defence)
            def_gens.append(def_gen)
        else:
            defences.append(None)
            def_gens.append(None)
    results["defence_name"] = defences
    results["def_gen"] = def_gens
    return results


def merge_attacks(results: pd.DataFrame):
    attacks = []
    for _, entry in results.iterrows():
        if "attack.init.name" in entry and entry["attack.init.name"] not in nones:
            attack = entry["attack.init.name"]
        else:
            attack = None
        attacks.append(attack)
    if attacks != [None] * len(attacks):
        results["attack_name"] = attacks
        results["atk_gen"] = [str(x).split(".")[-1] for x in attacks]
    return results


def parse_results(folder, files=["score_dict.json", "params.yaml"]):
    df = parse_folder(folder, files=files)
    df = flatten_results(df)
    df = merge_defences(df)
    df = merge_attacks(df)
    return df


def format_control_parameter(data, control_dict):
    logger.info("Formatting control parameters...")
    
    if hasattr(data, "def_gen"):
        defences = data.def_gen.unique()
    else:
        defences = []
    if hasattr(data, "atk_gen"):
        attacks = data.atk_gen.unique()
    else:
        attacks = []
    
    for defence in defences:
        if defence in control_dict:
            param = control_dict[defence]
            data.loc[data.def_gen == defence, "def_param"] = param.split(".")[-1]
            if param in data.columns:
                value = data[data.def_gen == defence][param]
            else:
                value = np.nan
            data.loc[data.def_gen == defence, "def_value"] = value
            control_dict.pop(defence)
        else:
            logger.warning(f"Defence {defence} not in control_dict. Deleting rows.")
            data = data[data.def_gen != defence]

    for attack in attacks:
        if attack in control_dict:
            param = control_dict[attack]
            data.loc[data.atk_gen == attack, "atk_param"] = param.split(".")[-1]
            if param in data.columns:
                value = data[data.atk_gen == attack][param]
            else:
                value = np.nan
            data.loc[data.atk_gen == attack, "atk_value"] = value
            control_dict.pop(attack)
        else:
            logger.warning(f"Attack {attack} not in control_dict. Deleting rows.")
            data = data[data.atk_gen != attack]

    return data


def clean_data_for_plotting(
    data,
    def_gen_dict,
    atk_gen_dict,
    control_dict,
    file,
    folder,
):
    logger.info("Replacing attack and defence names with short names...")
    if hasattr(data, "def_gen"):
        data.def_gen.fillna("Control", inplace=True)
        def_gen = data.def_gen.map(def_gen_dict)
        data.def_gen = def_gen
    if hasattr(data, "atk_gen"):
        atk_gen = data.atk_gen.map(atk_gen_dict)
        data.atk_gen = atk_gen
    logger.info("Dropping poorly merged columns...")
    data.dropna(axis=1, how="all", inplace=True)
    logger.info("Shortening model names...")
    data["model_name"] = data["model.init.name"].str.split(".").str[-1]
    data["model_layers"] = data["model_name"].str.split("Net").str[-1]
    # Rename columns that end in '.1' by removing the '.1'
    data.columns.rename(lambda x: x[:-2] if x.endswith(".1") else x, inplace=True)
    logger.info("Replacing data.sample.random_state with random_state...")
    data["data.sample.random_state"].rename("random_state", inplace=True)
    data = format_control_parameter(data, control_dict)
    logger.info(f"Saving data to {Path(folder) / file}")
    data.to_csv(Path(folder) / "data.csv")
    return data


def save_results(results, results_file) -> str:
    """
    Compile results from a folder of reports and save to a csv file; return the path to the csv file. It will optionally delete columns from the results.
    """
    logger.info("Compiling results...")

    suffix = Path(results_file).suffix
    if suffix == ".csv":
        results.to_csv(results_file)
    elif suffix == ".xlsx":
        results.to_excel(results_file)
    elif suffix == ".html":
        results.to_html(results_file)
    elif suffix == ".json":
        results.to_json(results_file)
    else:
        raise ValueError(f"File type {suffix} not supported.")
    assert Path(
        results_file,
    ).exists(), f"Results file {results_file} does not exist. Something went wrong."
    logger.debug(f"Results saved to {results_file}")
    return results_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="results.csv")
    parser.add_argument("--report_folder", type=str, default="reports", required=True)
    parser.add_argument("--config", type=str, default="conf/compile.yaml")
    parser.add_argument("--exclude", type=list, default=None, nargs="*")
    parser.add_argument("--verbose", type=str, default="INFO")
    parser.add_argument(
        "--kwargs",
        type=list,
        default=None,
        nargs="*",
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.verbose)
    report_folder = args.report_folder
    results_file = args.results_file
    results = parse_results(report_folder)
    with open(Path(Path(), args.config), "r") as f:
        big_dict = yaml.load(f, Loader=yaml.FullLoader)
    def_gen_dict = big_dict["defences"]
    atk_gen_dict = big_dict["attacks"]
    control_dict = big_dict["params"]
    results = clean_data_for_plotting(
        results,
        def_gen_dict,
        atk_gen_dict,
        control_dict,
        results_file,
        report_folder,
    )
    report_file = save_results(results, results_file)
    assert Path(
        report_file,
    ).exists(), f"Results file {report_file} does not exist. Something went wrong."
