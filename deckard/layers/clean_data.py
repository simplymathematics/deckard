import argparse
import logging
from pathlib import Path
from paretoset import paretoset
import pandas as pd
import seaborn as sns
import yaml
from math import isnan
import numpy as np
from .utils import deckard_nones as nones
from tqdm import tqdm
from .compile import save_results

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.8, font="times new roman")



def drop_frames_without_results(
    data,
    subset=[
        "accuracy",
        "adv_accuracy",
        "train_time",
        "adv_fit_time",
        "predict_time",
        "adv_success",
    ],
):
    logger.info(f"Dropping frames without results for {subset}")
    data.dropna(axis=0, subset=subset, inplace=True)
    return data

def calculate_failure_rate(data):
    logger.info("Calculating failure rate")
    data = data[data.columns.drop(list(data.filter(regex=r"\.1$")))]
    data.columns.str.replace(" ", "")
    if hasattr(data, "predict_time"):
        data.loc[:, "failure_rate"] = (
            (1 - data.loc[:, "accuracy"])
            * data.loc[:, "attack.attack_size"]
            / data.loc[:, "predict_time"]
        )
    elif hasattr(data, "predict_proba_time"):
        data.loc[:, "failure_rate"] = (
            (1 - data.loc[:, "accuracy"])
            * data.loc[:, "attack.attack_size"]
            / data.loc[:, "predict_proba_time"]
        )
    else:
        raise ValueError(
            "Data does not have predict_time or predict_proba_time as a column.",
        )
    data.loc[:, "adv_failure_rate"] = (
        (1 - data.loc[:, "adv_accuracy"])
        * data.loc[:, "attack.attack_size"]
        / data.loc[:, "adv_fit_time"]
    )

    data.loc[:, "training_time_per_failure"] = (
        data.loc[:, "train_time"] / data.loc[:, "failure_rate"]
    )

    data.loc[:, "training_time_per_adv_failure"] = (
        data.loc[:, "train_time_per_sample"] * data.loc[:, "adv_failure_rate"]
    )

    data.loc[:, "adv_training_time_per_failure"] = (
        data.loc[:, "train_time_per_sample"] * data.loc[:, "adv_failure_rate"]
    )
    return data


def pareto_set(data, sense_dict):
    new_sense_dict = {}
    for k, v in sense_dict.items():
        if k in data.columns:
            new_sense_dict[k] = v
        else:
            logger.warning(f"Column {k} not in data. Ignoring.")
    subset = data.loc[:, new_sense_dict.keys()]
    these = paretoset(subset, sense=new_sense_dict.values())
    return data.iloc[these, :]


def find_subset(data, **kwargs):
    if len(kwargs) > 0:
        qry = " and ".join(["{} == '{}'".format(k, v) for k, v in kwargs.items()])
        data.query(qry)
    return data


def min_max_scaling(data, *args):
    if "atk_gen" not in data.columns:
        attacks = []
    else:
        attacks = data.atk_gen.unique()
    if "def_gen" not in data.columns:
        defences = []
    else:
        defences = data.def_gen.unique()
    # Min-max scaling of control parameters
    for def_ in defences:
        max_ = data[data.def_gen == def_].def_value.max()
        min_ = data[data.def_gen == def_].def_value.min()
        scaled_value = (data[data.def_gen == def_].def_value - min_) / (max_ - min_)
        data.loc[data.def_gen == def_, "def_value"] = scaled_value
    for atk in attacks:
        max_ = data[data.atk_gen == atk].atk_value.max()
        min_ = data[data.atk_gen == atk].atk_value.min()
        scaled_value = (data[data.atk_gen == atk].atk_value - min_) / (max_ - min_)
        data.loc[data.atk_gen == atk, "atk_value"] = scaled_value
    for k in args:
        max_ = data[k].max()
        min_ = data[k].min()
        scaled_value = (data[k] - min_) / (max_ - min_)
        data[k] = scaled_value
    return data


def merge_defences(results: pd.DataFrame):
    defences = []
    def_gens = []
    defence_columns = [
        "model.art.pipeline.preprocessor.name",
        "model.art.pipeline.postprocessor.name",
        "model.art.pipeline.transformer.name",
        "model.art.pipeline.trainer.name",
    ]
    control_variable = [
        "model_layers",
    ]
    defaults = {
        "model.trainer.nb_epoch": 20,
        "model.trainer.kwargs.nb_epoch": 20,
    }
    assert "model_layers" in results.columns, "model_layers not in results.columns"
    for _, entry in tqdm(results.iterrows(), desc="Merging defences"):
        defence = []
        i = 0
        for col in defence_columns:
            if (
                col in entry
                and entry[col] not in nones
                
            ):
                defence.append(entry[col])
            else:
                pass
            i += 1
        for k, v in defaults.items():
            if (
                k in entry 
                and v != entry[k] 
                and not isnan(pd.to_numeric(entry[k]))
                and len(defence) == 0
            ):
                defence.append(k)
            else:
                pass
        for col in control_variable:
            if (
                col in entry
                and entry[col] not in nones
                and len(defence) == 0
            ):
                defence.append(col)
            else:
                pass
        ############################################################################################################
        if len(defence) > 1:
            def_gen = [str(x).split(".")[-1] for x in defence]
            defence = "_".join(defence)
            def_gen = "_".join(def_gen)
        elif len(defence) == 1 and defence[0] not in nones:
            def_gen = defence[0].split(".")[-1]
            defence = defence[0]
        else:
            def_gen = "Control"
            defence = "Control"
        ############################################################################################################
        defences.append(defence)
        def_gens.append(def_gen)
    results["defence_name"] = defences
    results["def_gen"] = def_gens
    logger.info(f"Unique defences after merging: {set(results.def_gen)}")
    assert hasattr(results,"def_gen"), "def_gen not in results.columns"
    return results


def merge_attacks(results: pd.DataFrame):
    attacks = []
    for _, entry in tqdm(results.iterrows(), desc="Merging attacks"):
        if "attack.init.name" in entry and entry["attack.init.name"] not in nones:
            attack = entry["attack.init.name"]
        else:
            attack = None
        attacks.append(attack)
    if attacks != [None] * len(attacks):
        results["attack_name"] = attacks
        results["atk_gen"] = [str(x).split(".")[-1] for x in attacks]
    logger.info(f"Unique attacks: {set(results.atk_gen)}")
    assert hasattr(results,"atk_gen"), "atk_gen not in results.columns"
    return results

def format_control_parameter(data, control_dict, fillna, def_control="model_layers"):
    logger.info("Formatting control parameters...")
    if "def_gen" in data:
        defences =  list(data.def_gen.unique())
    else:
        defences = []
    logger.info(f"Unique defences: {defences}")
    if "atk_gen" in data:
        attacks = list(data.atk_gen.unique())
    else:
        attacks = []
    logger.info(f"Unique attacks: {attacks}")
    logger.info("Fillna: ")
    logger.info(yaml.dump(fillna))
    for defence in defences:
        if defence in control_dict and defence != "Epochs":
            # Get parameter name from control_dict
            param = control_dict[defence]
            # Shorten parameter name
            data.loc[data.def_gen == defence, "def_param"] = param
            # Read parameter value from data if it exists, otherwise set to nan
            value = data[data.def_gen == defence][param] if param in data.columns else np.nan
            # strip whitespace
            value = value.str.strip() if isinstance(value, str) else value
            # Set value to numeric
            value = pd.to_numeric(value, errors="coerce")
            # Set value to data
            data.loc[data.def_gen == defence, "def_value"] = value
            logger.debug(f"Unique values for defence, {defence}:")
            logger.debug(f"{data[data.def_gen == defence].def_value.unique()}")
        elif defence in fillna.keys():
            param = control_dict[defence]
            value = data[data.def_gen == defence][param] if param in data.columns else np.nan
            value = pd.to_numeric(value, errors="coerce")
            value = value.fillna(fillna.get(defence, np.nan)) if isinstance(value, pd.Series) else value
            data.loc[data.def_gen == defence, "def_value"] = value
        else:
            logger.warning(f"Defence {defence} not in control_dict. Deleting rows.")
            data = data[data.def_gen != defence]
    for attack in attacks:
        if attack in control_dict:
            # Get parameter name from control_dict
            param = control_dict[attack]
            # Shorten parameter name
            data.loc[data.atk_gen == attack, "atk_param"] = param
            # Read parameter value from data if it exists, otherwise set to nan
            value = data[data.atk_gen == attack][param] if param in data.columns else np.nan
            # strip whitespace
            value = value.str.strip() if isinstance(value, str) else value
            # Set value to numeric
            value = pd.to_numeric(value, errors="coerce")
            # Fill nan values with fillna value
            value = value.fillna(fillna.get(attack, np.nan)) if isinstance(value, pd.Series) else value
            # Set value to data
            data.loc[data.atk_gen == attack, "atk_value"] = value
            logger.debug(f"Unique values for attack, {attack}:")
            logger.debug(f"{data[data.atk_gen == attack].atk_value.unique()}")
        else:
            logger.warning(f"Attack {attack} not in control_dict. Deleting rows.")
            data = data[data.atk_gen != attack]
    return data



def clean_data_for_plotting(
    data,
    def_gen_dict,
    atk_gen_dict,
    control_dict,
    fillna,

):
    
    
    logger.info(f"Dropping empty rows. Original shape: {data.shape}")
    data.dropna(axis=1, how="all", inplace=True)
    logger.info(f"Shape after dropping empty rows: {data.shape}")
    logger.info("Dropping poorly merged columns...")
    data = data.loc[:, ~data.columns.str.endswith(".1")]
    logger.info(f"Shape after dropping poorly merged columns: {data.shape}")
    logger.info("Shortening model names...")
    # Removes the path and to the model object and leaves the name of the model
    model_names = data["model.init.name"].str.split(".").str[-1]
    data["model_name"] = model_names
    # If "Net" is in the model name, we assume the following string denotes the layers as in ResNet18
    if hasattr(data, "model.init.name"):
        model_names = data["model.init.name"].str.split(".").str[-1]
        data.loc[:, "model_name"] = model_names
        model_layers = [str(x).split("Net")[-1] for x in model_names]
        data.loc[:, "model_layers"] = model_layers
        logger.info(f"Model Names: {data.model_name.unique()}")
        logger.info(f"Model Layers: {data.model_layers.unique()}")
    data['nb_epoch'] = data['model.trainer.kwargs.nb_epoch'] if "model.trainer.kwargs.nb_epoch" in data.columns else data['model.trainer.nb_epoch']
    logger.info("Replacing data.sample.random_state with random_state...")
    data["data.sample.random_state"].rename("random_state", inplace=True)
    data = merge_defences(data)
    
    logger.info("Replacing attack and defence names with short names...")
    if hasattr(data, "def_gen"):
        def_gen = data.def_gen.map(def_gen_dict)
        data.def_gen = def_gen
        data.dropna(axis=0, subset=["def_gen"], inplace=True)
    data = merge_attacks(data)
    if hasattr(data, "atk_gen"):
        atk_gen = data.atk_gen.map(atk_gen_dict)
        data.atk_gen = atk_gen
        data.dropna(axis=0, subset=["atk_gen"], inplace=True)
    data = format_control_parameter(data, control_dict, fillna)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="Data file to read from",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Data file to read from",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default="INFO",
        help="Increase output verbosity",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file",
        default="clean.yaml",
    )
    parser.add_argument(
        "-s",
        "--subset",
        help="Subset of data you would like to plot",
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "-d",
        "--drop_if_empty",
        help="Drop row if this columns is empty",
        nargs="+",
        type=str,
        default=[
            "accuracy",
            "adv_accuracy",
            "train_time",
            "adv_fit_time",
            "predict_proba_time",
        ],
    )
    parser.add_argument(
        "--pareto_dict",
        help="Path to (optional) pareto set dictionary.",
        default=None,
    )
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    assert Path(
        args.input_file,
    ).exists(), f"File {args.input_file} does not exist. Please specify a valid file using the -i flag."
    data = pd.read_csv(args.input_file)
    if isinstance(args.drop_if_empty, str):
        args.drop_if_empty = args.drop_if_empty.split(",")
    else:
        assert isinstance(args.drop_if_empty, list)
    data = drop_frames_without_results(
        data,
        subset=args.drop_if_empty,
    )
    if args.pareto_dict is None:
        sense_dict = {}
    else:
        if Path(args.pareto_dict).exists():
            with open(args.pareto_dict, "r") as f:
                sense_dict = yaml.safe_load(f)
        elif (
            isinstance(args.pareto_dict.split(":")[:-2], str)
            and Path(args.pareto_dict.split(":")[:-2]).exists()
        ):
            with open(Path(args.pareto_dict.split(":")[:-2]), "r") as f:
                sense_dict = yaml.safe_load(f)[args.pareto_dict.split(":")[:-1]]
        else:
            raise ValueError(
                f"Pareto_dictionary, {args.pareto_dict} does not exist as a file or file and dictionary using file:dictionary notation.",
            )
     # Reads Config file
    with open(Path(args.config), "r") as f:
        big_dict = yaml.load(f, Loader=yaml.FullLoader)
    def_gen_dict = big_dict["defences"]
    atk_gen_dict = big_dict["attacks"]
    control_dict = big_dict["params"]
    fillna = big_dict.get("fillna", {})
    min_max = big_dict.get("min_max", ["nb_epoch"])

    results = clean_data_for_plotting(
        data,
        def_gen_dict,
        atk_gen_dict,
        control_dict,
        fillna=fillna,
    )
    results = calculate_failure_rate(results)
    
    results = min_max_scaling(results, *min_max)
    output_file = save_results(results, Path(args.output_file).name, Path(args.output_file).parent)
    assert Path(output_file).exists(), f"File {output_file} does not exist. Please specify a valid file using the -o flag."
    logger.info(f"Saved results to {output_file}")
    
    

