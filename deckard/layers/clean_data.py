import argparse
import logging
from pathlib import Path
from paretoset import paretoset
import pandas as pd
import seaborn as sns
import yaml
from math import isnan
import numpy as np
from tqdm import tqdm

from .utils import deckard_nones as nones
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
    """
    The function `drop_frames_without_results` drops rows from a DataFrame that have missing values in
    specified columns.

    Args:
      data: The `data` parameter is a pandas DataFrame that contains the results of some experiment or
    analysis.
      subset: The `subset` parameter is a list of column names that are used to determine whether a
    frame has results or not. If any of the columns specified in the `subset` list have missing values
    (NaN), then the corresponding frame will be dropped from the `data` DataFrame.

    Returns:
      the modified DataFrame after dropping the frames without results.
    """

    logger.info(f"Dropping frames without results for {subset}")
    data.dropna(axis=0, subset=subset, inplace=True)
    return data


def calculate_failure_rate(data):
    """
    The function `calculate_failure_rate` calculates failure rates and training times based on input
    data.

    Args:
      data: The `data` parameter is expected to be a pandas DataFrame containing the following columns:

    Returns:
      the modified "data" dataframe with additional columns added, including "adv_failure_rate",
    "failure_rate", "training_time_per_failure", and "training_time_per_adv_failure".
    """
    logger.info("Calculating failure rate")
    data = data[data.columns.drop(list(data.filter(regex=r"\.1$")))]
    data.columns.str.replace(" ", "")
    assert "accuracy" in data.columns, "accuracy not in data.columns"
    data.loc[:, "accuracy"] = pd.to_numeric(data.loc[:, "accuracy"])
    assert (
        "attack.attack_size" in data.columns
    ), "attack.attack_size not in data.columns"
    data.loc[:, "attack.attack_size"] = pd.to_numeric(
        data.loc[:, "attack.attack_size"],
    )
    assert (
        "predict_time" in data.columns or "predict_proba_time" in data.columns
    ), "predict_time or predict_proba_time not in data.columns"
    assert "adv_accuracy" in data.columns, "adv_accuracy not in data.columns"
    data.loc[:, "adv_accuracy"] = pd.to_numeric(data.loc[:, "adv_accuracy"])
    assert "adv_fit_time" in data.columns, "adv_fit_time not in data.columns"
    data.loc[:, "adv_fit_time"] = pd.to_numeric(data.loc[:, "adv_fit_time"])
    assert "train_time" in data.columns, "train_time not in data.columns"
    data.loc[:, "train_time"] = pd.to_numeric(data.loc[:, "train_time"])
    if "predict_time" in data.columns:
        data.loc[:, "predict_time"] = pd.to_numeric(data.loc[:, "predict_time"])
        failure_rate = (
            (1 - data.loc[:, "accuracy"]) * data.loc[:, "attack.attack_size"]
        ) / data.loc[:, "predict_time"]
    elif "predict_proba_time" in data.columns:
        data.loc[:, "predict_proba_time"] = pd.to_numeric(
            data.loc[:, "predict_proba_time"],
        )
        failure_rate = (
            (1 - data.loc[:, "accuracy"]) * data.loc[:, "attack.attack_size"]
        ) / data.loc[:, "predict_proba_time"]
    else:
        raise ValueError("predict_time or predict_proba_time not in data.columns")
    adv_failure_rate = (
        (1 - data.loc[:, "adv_accuracy"])
        * data.loc[:, "attack.attack_size"]
        / data.loc[:, "predict_time"]
    )

    data = data.assign(adv_failure_rate=adv_failure_rate)
    data = data.assign(failure_rate=failure_rate)
    training_time_per_failure = data.loc[:, "train_time"] / data.loc[:, "failure_rate"]
    training_time_per_adv_failure = (
        data.loc[:, "train_time"] * data.loc[:, "adv_failure_rate"]
    )
    data = data.assign(training_time_per_failure=training_time_per_failure)
    data = data.assign(training_time_per_adv_failure=training_time_per_adv_failure)

    assert (
        "training_time_per_adv_failure" in data.columns
    ), "training_time_per_adv_failure not in data.columns"
    assert (
        "training_time_per_failure" in data.columns
    ), "training_time_per_failure not in data.columns"
    assert "adv_failure_rate" in data.columns, "adv_failure_rate not in data.columns"
    assert "failure_rate" in data.columns, "failure_rate not in data.columns"
    return data


def pareto_set(data, sense_dict):
    """
    The function `pareto_set` takes in a dataset and a dictionary specifying the sense (minimize or
    maximize) for each column, and returns the subset of the dataset that represents the Pareto set.

    Args:
      data: The `data` parameter is a pandas DataFrame that contains the data you want to analyze. It
    should have columns that correspond to the keys in the `sense_dict`.
      sense_dict: The `sense_dict` parameter is a dictionary that maps column names in the `data`
    DataFrame to their corresponding sense (either "min" or "max"). This is used to specify the
    optimization direction for each column when finding the Pareto set.

    Returns:
      a subset of the input data that represents the Pareto set.
    """
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
    """
    The function `find_subset` takes a dataset and optional keyword arguments, and returns a subset of
    the dataset based on the provided arguments.

    Args:
      data: The `data` parameter is a pandas DataFrame that contains the dataset you want to filter.

    Returns:
      the input data, possibly filtered based on the keyword arguments provided.
    """
    if len(kwargs) > 0:
        qry = " and ".join(["{} == '{}'".format(k, v) for k, v in kwargs.items()])
        data.query(qry)
    return data


def min_max_scaling(data, *args):
    """
    The function `min_max_scaling` performs min-max scaling on specified columns of a given dataset.

    Args:
      data: The `data` parameter is a pandas DataFrame that contains the data you want to perform
    min-max scaling on. It should have columns named "atk_gen" and "def_gen" for attack and defense
    generations, and columns named "atk_value" and "def_value" for attack and defense values.

    Returns:
      the modified data with the control parameters scaled using min-max scaling.
    """
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
        max_ = pd.to_numeric(max_, errors="raise")
        min_ = data[data.def_gen == def_].def_value.min()
        min_ = pd.to_numeric(min_, errors="raise")
        scaled_value = (data[data.def_gen == def_].def_value - min_) / (max_ - min_)
        data.loc[data.def_gen == def_, "def_value"] = scaled_value
    for atk in attacks:
        max_ = data[data.atk_gen == atk].atk_value.max()
        max_ = pd.to_numeric(max_, errors="raise")
        min_ = data[data.atk_gen == atk].atk_value.min()
        min_ = pd.to_numeric(min_, errors="raise")
        scaled_value = (data[data.atk_gen == atk].atk_value - min_) / (max_ - min_)
        data.loc[data.atk_gen == atk, "atk_value"] = scaled_value
    for k in args:
        max_ = data[k].max()
        max_ = pd.to_numeric(max_, errors="raise")
        min_ = data[k].min()
        min_ = pd.to_numeric(min_, errors="raise")
        data[k] = pd.to_numeric(data[k], errors="raise")
        scaled_value = (data[k] - min_) / (max_ - min_)
        data[k] = scaled_value
    return data


def merge_defences(
    results: pd.DataFrame,
    defence_columns=[
        "model.art.pipeline.preprocessor.name",
        "model.art.pipeline.postprocessor.name",
        "model.art.pipeline.transformer.name",
        "model.art.pipeline.trainer.name",
        "model.art.preprocessor.name",
        "model.art.postprocessor.name",
        "model.art.transformer.name",
        "model.art.trainer.name",
    ],
    control_variable=["model_layers"],
    defaults={
        "model.trainer.nb_epoch": 20,
    },
):
    """
    The function `merge_defences` merges different defence columns in a DataFrame and assigns a unique
    name to each merged defence.

    Args:
      results (pd.DataFrame): A pandas DataFrame containing the results of different defences.
      defence_columns: A list of column names in the `results` DataFrame that represent the different
    components of a defence. These columns will be used to merge the defences.
      control_variable: The `control_variable` parameter is a list of column names in the `results`
    DataFrame that represent control variables. These variables are used to group the results and create
    a separate defense for each unique combination of control variables.
      defaults: The `defaults` parameter is a dictionary that contains default values for certain
    parameters. These default values will be used if the corresponding parameter is not present in the
    `results` DataFrame.

    Returns:
      the modified `results` DataFrame with two additional columns: `defence_name` and `def_gen`.
    """
    defences = []
    def_gens = []
    for control in control_variable:
        assert control in results, f"{control} not in results.columns"
    for _, entry in tqdm(results.iterrows(), desc="Merging defences"):
        defence = []
        i = 0
        for col in defence_columns:
            if col in entry and entry[col] not in nones:
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
            if col in entry and entry[col] not in nones and len(defence) == 0:
                defence.append(col)
            else:
                pass
        ############################################################################################################
        if len(defence) > 1:
            def_gen = [str(x).split(".")[-1] for x in defence]
            defence = "_".join(defence)
            def_gen = "_".join(def_gen)
        elif len(defence) == 1 and defence[0] not in nones and defence[0] != np.nan:
            def_gen = str(defence[0]).split(".")[-1]
            defence = str(defence[0])
        else:
            def_gen = "Control"
            defence = "Control"
        ############################################################################################################
        defences.append(defence)
        def_gens.append(def_gen)
    results["defence_name"] = defences
    results["def_gen"] = def_gens
    logger.info(f"Unique defences after merging: {set(results.def_gen)}")
    logger.info(f"Unique set of full names after merge: {set(results.defence_name)}")
    assert hasattr(results, "def_gen"), "def_gen not in results.columns"
    return results


def merge_attacks(results: pd.DataFrame):
    """
    The function `merge_attacks` merges attack information from a DataFrame and adds it to the DataFrame
    as new columns.

    Args:
      results (pd.DataFrame): The `results` parameter is expected to be a pandas DataFrame containing
    the results of some analysis or computation. It is assumed that the DataFrame has a column named
    "attack.init.name" which contains the names of attacks. The function iterates over each row of the
    DataFrame, checks if the "attack.init

    Returns:
      the modified `results` DataFrame with additional columns `attack_name` and `atk_gen`.
    """
    attacks = []
    for _, entry in tqdm(results.iterrows(), desc="Merging attacks"):
        if "attack.init.name" in entry and entry["attack.init.name"] not in nones:
            attack = entry["attack.init.name"]
        else:
            attack = None
        attacks.append(attack)
    if attacks != [None] * len(attacks):
        results = results.assign(attack_name=attacks)
        attacks = [str(x).split(".")[-1] for x in attacks]
        results = results.assign(atk_gen=attacks)
        logger.info(f"Unique attacks: {set(results.atk_gen)}")
    else:
        logger.warning("No attacks found in data. Check your config file.")
    assert hasattr(results, "atk_gen"), "atk_gen not in results.columns"
    return results


def format_control_parameter(data, control_dict, fillna):
    """
    The function `format_control_parameter` takes in data, a control dictionary, and a fillna parameter,
    and formats the control parameters in the data based on the control dictionary and fillna values.

    Args:
      data: The `data` parameter is a pandas DataFrame that contains the control parameters for a
    system. It should have columns named "def_gen" and "atk_gen" which represent the defence and attack
    generators respectively. The DataFrame may also contain additional columns that represent the
    control parameters for each defence and attack.
      control_dict: The `control_dict` parameter is a dictionary that maps defence and attack names to
    their corresponding parameter names. It is used to retrieve the parameter name for each defence and
    attack in the `data` dataframe.
      fillna: The `fillna` parameter is a dictionary that contains default values to fill missing values
    in the data. The keys of the dictionary correspond to the names of defences or attacks, and the
    values are the default values to be used for filling missing values.

    Returns:
      the modified "data" dataframe with additional columns "def_param", "atk_param", "def_value", and
    "atk_value".
    """
    logger.info("Formatting control parameters...")
    if "def_gen" in data:
        defences = list(data.def_gen.unique())
    else:
        defences = []

    if "atk_gen" in data:
        attacks = list(data.atk_gen.unique())
    else:
        attacks = []
    logger.info(f"Unique defences: {defences}")
    logger.info(f"Unique attacks: {attacks}")
    logger.info("Fillna: ")
    logger.info(yaml.dump(fillna))
    for defence in defences:
        if defence in control_dict:
            # Get parameter name from control_dict
            param = control_dict[defence]
            # Shorten parameter name
            data.loc[data.def_gen == defence, "def_param"] = param
            # Read parameter value from data if it exists, otherwise set to nan
            value = (
                data[data.def_gen == defence][param]
                if param in data.columns
                else np.nan
            )
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
            value = (
                data[data.def_gen == defence][param]
                if param in data.columns
                else np.nan
            )
            value = pd.to_numeric(value, errors="coerce")
            value = (
                value.fillna(fillna.get(defence, np.nan))
                if isinstance(value, pd.Series)
                else value
            )
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
            value = (
                data[data.atk_gen == attack][param] if param in data.columns else np.nan
            )
            # strip whitespace
            value = value.str.strip() if isinstance(value, str) else value
            # Set value to numeric
            value = pd.to_numeric(value, errors="coerce")
            # Fill nan values with fillna value
            value = (
                value.fillna(fillna.get(attack, np.nan))
                if isinstance(value, pd.Series)
                else value
            )
            # Set value to data
            data.loc[data.atk_gen == attack, "atk_value"] = value
            logger.debug(f"Unique values for attack, {attack}:")
            logger.debug(f"{data[data.atk_gen == attack].atk_value.unique()}")
        elif attack in fillna.keys():
            param = control_dict[attack]
            value = (
                data[data.atk_gen == attack][param] if param in data.columns else np.nan
            )
            value = pd.to_numeric(value, errors="coerce")
            value = (
                value.fillna(fillna.get(attack, np.nan))
                if isinstance(value, pd.Series)
                else value
            )
            data.loc[data.atk_gen == attack, "def_value"] = value
        else:
            logger.warning(f"Attack {attack} not in control_dict. Deleting rows.")
            data = data[data.atk_gen != attack]
    defences = list(data.def_gen.unique())
    attacks = list(data.atk_gen.unique())
    logger.info(f"Unique defences: {defences}")
    logger.info(f"Unique attacks: {attacks}")
    assert "def_param" in data.columns, "def_param not in data.columns"
    assert "atk_param" in data.columns, "atk_param not in data.columns"
    assert "def_value" in data.columns, "def_value not in data.columns"
    assert "atk_value" in data.columns, "atk_value not in data.columns"
    return data


def clean_data_for_plotting(
    data,
    def_gen_dict,
    atk_gen_dict,
    control_dict,
    fillna,
):
    """
    The function `clean_data_for_plotting` cleans and formats data for plotting by dropping empty rows,
    removing poorly merged columns, shortening model names, replacing certain column names, and
    formatting control parameters.

    Args:
      data: The `data` parameter is a pandas DataFrame that contains the data to be cleaned and prepared
    for plotting.
      def_gen_dict: The `def_gen_dict` parameter is a dictionary that maps the names of defence
    generators to their corresponding short names. It is used to replace the defence generator names in
    the data with their short names.
      atk_gen_dict: The `atk_gen_dict` parameter is a dictionary that maps the attack names in the data
    to their corresponding short names. It is used to replace the attack names with their short names in
    the cleaned data.
      control_dict: The `control_dict` parameter is a dictionary that contains control parameters for
    formatting the data. It is used in the `format_control_parameter` function.
      fillna: The `fillna` parameter is used to specify how missing values in the data should be filled.
    It is a value or a dictionary of values where the keys are column names and the values are the
    values to fill in for missing values in those columns.

    Returns:
      the cleaned and formatted data.
    """
    logger.info(f"Dropping empty rows. Original shape: {data.shape}")
    data.dropna(axis=1, how="all", inplace=True)
    logger.info(f"Shape after dropping empty rows: {data.shape}")
    logger.info("Dropping poorly merged columns...")
    data = data.loc[:, ~data.columns.str.endswith(".1")]
    logger.info(f"Shape after dropping poorly merged columns: {data.shape}")
    logger.info("Shortening model names...")
    # If "Net" is in the model name, we assume the following string denotes the layers as in ResNet18
    if hasattr(data, "model.init.name"):
        model_names = data["model.init.name"].str.split(".").str[-1]
        data = data.assign(model_name=model_names)
        model_layers = [str(x).split("Net")[-1] for x in model_names]
        data = data.assign(model_layers=model_layers)
        logger.info(f"Model Names: {data.model_name.unique()}")
        logger.info(f"Model Layers: {data.model_layers.unique()}")

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
    # Strip whitespace from column names
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x  # noqa E731
    data.rename(columns=trim_strings, inplace=True)
    # Strip whitespace from column values
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    assert "model.init.name" in data.columns, "model.init.name not in data.columns"

    if isinstance(args.drop_if_empty, str):
        args.drop_if_empty = args.drop_if_empty.split(",")
    else:
        assert isinstance(args.drop_if_empty, list)
    for col in args.drop_if_empty:
        assert col in data.columns, f"Column {col} not in data.columns"
    data = drop_frames_without_results(data, subset=args.drop_if_empty)
    if args.pareto_dict is None:
        sense_dict = {}
    else:
        if Path(args.pareto_dict).exists():
            with open(args.pareto_dict, "r") as f:
                sense_dict = yaml.safe_load(f)
        elif (
            isinstance(args.pareto_dict.split(":")[:-1], str)
            and Path(args.pareto_dict.split(":")[:-2]).exists()
        ):
            with open(Path(args.pareto_dict.split(":")[:-1]), "r") as f:
                sense_dict = yaml.safe_load(f)[args.pareto_dict.split(":")[:-1]]
        else:
            raise ValueError(
                f"Pareto_dictionary, {args.pareto_dict} does not exist as a file or file and dictionary using file:dictionary notation.",
            )
    # Reads Config file
    with open(Path(args.config), "r") as f:
        big_dict = yaml.load(f, Loader=yaml.FullLoader)
    def_gen_dict = big_dict.get("defences", {})
    atk_gen_dict = big_dict.get("attacks", {})
    control_dict = big_dict.get("params", {})
    fillna = big_dict.get("fillna", {})
    min_max = big_dict.get("min_max", [])

    results = clean_data_for_plotting(
        data,
        def_gen_dict,
        atk_gen_dict,
        control_dict,
        fillna=fillna,
    )
    results = calculate_failure_rate(results)

    results = min_max_scaling(results, *min_max)
    output_file = save_results(
        results,
        Path(args.output_file).name,
        Path(args.output_file).parent,
    )
    assert Path(
        output_file,
    ).exists(), f"File {output_file} does not exist. Please specify a valid file using the -o flag."
    logger.info(f"Saved results to {output_file}")
