import argparse
import logging
from pathlib import Path
from paretoset import paretoset
import pandas as pd
import seaborn as sns
import yaml
import numpy as np
from tqdm import tqdm
from .utils import deckard_nones as nones
from .compile import save_results, load_results

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.8, font="times new roman")


def fill_train_time(
    data,
    match=[
        "model.init.name",
        "model.trainer.nb_epoch",
        "model.art.preprocessor",
        "model.art.postprocessor",
        "def_name",
        "def_gen",
        "defence_name",
    ],
):
    sort_by = []
    for col in match:
        # find out which columns have the string in match
        if col in data.columns:
            sort_by.append(col)
        else:
            pass
    # Convert "train_time" to numeric
    # Fill missing values in "train_time" with the max of the group
    data["train_time"] = pd.to_numeric(data["train_time"], errors="coerce").astype(
        float,
    )
    # Group by everything in the "match" list
    assert isinstance(data, pd.DataFrame), "data is not a pandas DataFrame"
    # Sort by the entries in "match"
    data = data.sort_values(by=sort_by)
    data["train_time"] = data["train_time"].fillna(method="ffill")
    data["train_time"] = data["train_time"].fillna(method="bfill")
    return data


def drop_rows_without_results(
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
    The function `drop_rows_without_results` drops rows from a DataFrame that have missing values in
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

    for col in subset:
        logger.info(f"Dropping frames without results for {col}")
        before = data.shape[0]
        if col not in data.columns:
            logger.warning(f"{col} not in data.columns. Ignoring.")
            continue
        data.dropna(axis=0, subset=[col], inplace=True)
        after = data.shape[0]
        logger.info(f"Shape of data after data after dropping na: {data.shape}")
        percent_change = (before - after) / before * 100
        if percent_change > 5:
            # input(f"{percent_change:.2f}% of data dropped for {col}. Press any key to continue.")
            logger.warning(f"{percent_change:.2f}% of data dropped for {col}")
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
        survival_time = data.loc[:, "predict_time"] * data.loc[:, "accuracy"]
    elif "predict_proba_time" in data.columns:
        data.loc[:, "predict_proba_time"] = pd.to_numeric(
            data.loc[:, "predict_proba_time"],
        )
        failure_rate = (
            (1 - data.loc[:, "accuracy"]) * data.loc[:, "attack.attack_size"]
        ) / data.loc[:, "predict_proba_time"]
    else:
        raise ValueError("predict_time or predict_proba_time not in data.columns")
    if "adv_fit_time" in data.columns:
        assert "adv_accuracy" in data.columns, "adv_accuracy not in data.columns"
        if "predict_time" in data.columns:
            adv_failure_rate = (
                (1 - data.loc[:, "adv_accuracy"])
                * data.loc[:, "attack.attack_size"]
                / data.loc[:, "adv_fit_time"]
            )
            adv_survival_time = (
                data.loc[:, "predict_time"] * data.loc[:, "adv_accuracy"]
            )
        elif "predict_proba_time" in data.columns:
            adv_failure_rate = (
                (1 - data.loc[:, "adv_accuracy"])
                * data.loc[:, "attack.attack_size"]
                / data.loc[:, "adv_fit_time"]
            )
            adv_survival_time = (
                data.loc[:, "predict_proba_time"] * data.loc[:, "adv_accuracy"]
            )
        else:
            raise ValueError("predict_time or predict_proba_time not in data.columns")
    data = data.assign(adv_survival_time=adv_survival_time)
    data = data.assign(survival_time=survival_time)
    data = data.assign(adv_failure_rate=adv_failure_rate)
    data = data.assign(failure_rate=failure_rate)
    training_time_per_failure = data.loc[:, "train_time"] / data.loc[:, "survival_time"]
    training_time_per_adv_failure = (
        data.loc[:, "train_time"] * data.loc[:, "adv_survival_time"]
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
    for _, entry in tqdm(results.iterrows(), desc="Merging defences"):
        defence = []
        i = 0
        # Explicit defences from ART
        for col in defence_columns:
            if col in entry and entry[col] not in nones:
                defence.append(entry[col])
            else:
                pass
            i += 1
        ############################################################################################################
        if len(defence) > 1:
            def_gen = [str(str(x).split(".")[-1]) for x in defence]
            defence = "_".join(str(defence))
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
            logger.info(f"Unique values for defence, {defence}:")
            logger.info(f"{data[data.def_gen == defence].def_value.unique()}")
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
            del fillna[defence]
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
            logger.info(f"Unique values for attack, {attack}:")
            logger.info(f"{data[data.atk_gen == attack].atk_value.unique()}")
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
            del fillna[attack]
        else:
            logger.warning(f"Attack {attack} not in control_dict. Deleting rows.")
            data = data[data.atk_gen != attack]
    defences = list(data.def_gen.unique()) if "def_gen" in data.columns else []
    attacks = list(data.atk_gen.unique()) if "atk_gen" in data.columns else []
    logger.info(f"Unique defences: {defences}")
    logger.info(f"Unique attacks: {attacks}")
    if len(defences) > 0:
        assert "def_param" in data.columns, "def_param not in data.columns"
        assert "def_value" in data.columns, "def_value not in data.columns"
    if len(attacks) > 0:
        assert "atk_param" in data.columns, "atk_param not in data.columns"
        assert "atk_value" in data.columns, "atk_value not in data.columns"
    return data, fillna


def replace_strings_in_data(data, replace_dict):
    for k, v in replace_dict.items():
        logger.info(f"Replacing strings in {k}...")
        assert isinstance(
            v,
            dict,
        ), f"Value for key {k} in replace_dict is not a dictionary."
        if k not in data.columns:
            logger.warning(f"Column {k} not in data. Ignoring.")
            continue
        for k1, v1 in v.items():
            logger.info(f"Replacing {k1} with {v1} in {k}...")
            k1 = str(k1)
            v1 = str(v1)
            data[k] = data[k].astype(str)
            data.loc[:, k] = data.loc[:, k].str.replace(k1, v1)
        logger.info(f"Unique values after replacement: {data[k].unique()}")
    return data


def replace_strings_in_columns(data, replace_dict):
    cols = list(data.columns)
    for k, v in replace_dict.items():
        logger.info(f"Replacing {k} with {v} in column names...")
        for col in cols:
            if k == col:
                logger.info(f"Replacing {k} with {v} in column names...")
                data.rename(columns={k: v}, inplace=True)
            else:
                pass
    after = list(data.columns)
    if len(replace_dict) > 0:
        logger.info(f"Columns after replacement: {after}")
        assert cols != after, "Columns not replaced."
        assert len(cols) == len(
            after,
        ), f"Length of columns before and after replacement not equal: {len(cols)} != {len(after)}."
    return data


def clean_data_for_plotting(
    data,
    def_gen_dict={},
    atk_gen_dict={},
    control_dict={},
    fillna={},
    replace_dict={},
    col_replace_dict={},
    pareto_dict={},
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
    if len(def_gen_dict) > 0:
        data = merge_defences(data)
    logger.info("Replacing attack and defence names with short names...")
    if hasattr(data, "def_gen"):
        shorten_defence_names(data, def_gen_dict)
    if "attack.init.name" in data:
        data = merge_attacks(data)
    if hasattr(data, "atk_gen"):
        shorten_attack_names(data, atk_gen_dict)
    data, fillna = format_control_parameter(data, control_dict, fillna)
    data = fill_na(data, fillna)
    data = replace_strings_in_data(data, replace_dict)
    data = replace_strings_in_columns(data, col_replace_dict)
    if len(pareto_dict) > 0:
        data = find_pareto_set(data, pareto_dict)
    return data


def shorten_defence_names(data, def_gen_dict):
    def_gen = data.def_gen.map(def_gen_dict)
    data.def_gen = def_gen
    data.dropna(axis=0, subset=["def_gen"], inplace=True)


def shorten_attack_names(data, atk_gen_dict):
    atk_gen = data.atk_gen.map(atk_gen_dict)
    data.atk_gen = atk_gen
    data.dropna(axis=0, subset=["atk_gen"], inplace=True)


def fill_na(data, fillna):
    for k, v in fillna.items():
        if k in data.columns:
            data[k] = data[k].fillna(v)
        else:
            data[k] = str(v)
    return data


def find_pareto_set(data, pareto_dict):
    data = pareto_set(data, pareto_dict)
    return data


def drop_values(data, drop_dict):
    for k, v in drop_dict.items():
        data = data[data[k] != v]
    return data


clean_data_parser = argparse.ArgumentParser()
clean_data_parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    help="Data file to read from",
    required=True,
)
clean_data_parser.add_argument(
    "-o",
    "--output_file",
    type=str,
    help="Data file to read from",
    required=True,
)
clean_data_parser.add_argument(
    "-v",
    "--verbosity",
    default="INFO",
    help="Increase output verbosity",
)
clean_data_parser.add_argument(
    "-c",
    "--config",
    help="Path to the config file",
    default="clean.yaml",
)
clean_data_parser.add_argument(
    "-s",
    "--subset",
    help="Subset of data you would like to plot",
    default=None,
    nargs="?",
)
clean_data_parser.add_argument(
    "-d",
    "--drop_if_empty",
    help="Drop row if this columns is empty",
    nargs="+",
    type=str,
    default=[
        "accuracy",
        "train_time",
        "predict_time",
    ],
)
clean_data_parser.add_argument(
    "--pareto_dict",
    help="Path to (optional) pareto set dictionary.",
    default=None,
)


def clean_data_main(args):
    logging.basicConfig(level=args.verbosity)
    assert Path(
        args.input_file,
    ).exists(), f"File {args.input_file} does not exist. Please specify a valid file using the -i flag."
    data = load_results(
        results_file=Path(args.input_file).name,
        results_folder=Path(args.input_file).parent,
    )
    # Strip whitespace from column names
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x  # noqa E731
    data.rename(columns=trim_strings, inplace=True)
    # Strip whitespace from column values
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    if isinstance(args.drop_if_empty, str):
        args.drop_if_empty = args.drop_if_empty.split(",")
    else:
        assert isinstance(args.drop_if_empty, list)

    # Reads Config file
    with open(Path(args.config), "r") as f:
        big_dict = yaml.load(f, Loader=yaml.FullLoader)
    def_gen_dict = big_dict.get("defences", {})
    atk_gen_dict = big_dict.get("attacks", {})
    control_dict = big_dict.get("params", {})
    fillna = big_dict.get("fillna", {})
    min_max = big_dict.get("min_max", [])
    replace_dict = big_dict.get("replace", {})
    replace_cols = big_dict.get("replace_cols", {})
    pareto_dict = big_dict.get("pareto", {})
    drop_dict = big_dict.pop("drop_values", {})
    data = drop_values(data, drop_dict)
    results = clean_data_for_plotting(
        data,
        def_gen_dict,
        atk_gen_dict,
        control_dict,
        fillna=fillna,
        replace_dict=replace_dict,
        col_replace_dict=replace_cols,
        pareto_dict=pareto_dict,
    )
    for col in results.columns:
        if "def" in col:
            print(col)
    if "adv_accuracy" in results.columns:
        results = calculate_failure_rate(results)
    results = fill_train_time(results)
    results = drop_rows_without_results(results, subset=args.drop_if_empty)
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


if __name__ == "__main__":
    args = clean_data_parser.parse_args()
    clean_data_main(args)
