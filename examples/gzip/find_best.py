# Queries the optuna database to find the best trial for a given study.

import optuna
from pathlib import Path
import yaml
import argparse
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra import compose, initialize

logger = logging.getLogger(__name__)


def query_optuna_db(study_name, storage_path):
    # Load the study
    logger.info(f"Loading study: {study_name} from {storage_path}")
    study = optuna.load_study(study_name=study_name, storage=storage_path)
    # Create a dataframe from the trials
    trials = study.trials_dataframe()
    cols = trials.columns
    new_cols = []
    id_vars = []
    value_vars = []
    for col in cols:
        # Check for hydra overrides and remove them for clarity
        if col.startswith("++"):
            col = col[2:]
        elif col.startswith("+") or col.startswith("~"):
            col = col[1:]
        else:
            pass
        # Check the optuna params, user_attrs to group by
        if col.startswith("params_"):
            new_col = col[7:]
            id_vars.append(new_col)
        elif col.startswith("user_attrs_"):
            new_col = col[11:]
            id_vars.append(new_col)
        elif col.startswith("value_"):  # the optimization metric(s)
            new_col = col[6:]
            value_vars.append(new_col)
        else:
            new_col = col
        new_cols.append(new_col)
    trials.columns = new_cols
    logger.info(f"ID variables: {id_vars}")
    logger.info(f"Value variables: {value_vars}")
    # drop system_attrs_grid_id,system_attrs_search_space, duration, datatime_start, datetime_complete
    trials = trials.drop(
        columns=[
            "system_attrs_grid_id",
            "system_attrs_search_space",
            "duration",
            "datetime_start",
            "datetime_complete",
            "state",
            "number",
        ],
    )
    return trials, id_vars, value_vars


def find_mean_std(trials, id_vars, value_vars):
    # For each value_var, calculate the mean and std
    grouped = trials.groupby(id_vars)
    for var_ in tqdm(value_vars, desc="Calculating mean and std"):
        trials[f"avg_{var_}"] = grouped[var_].transform("mean")
        trials[f"std_{var_}"] = grouped[var_].transform(lambda X: X.std())
    return trials


def sort_by_value_vars(trials, value_vars):
    # sort by value_vars, each of which has an avg_${var} and a
    # std_${var} column
    # TODO: Add direction(s) as an argument
    sortby = []
    ascending = [False, True] * len(value_vars)
    for var_ in value_vars:
        # Assuming you want to maximize the average value and minimize the std (as a tie breaker)
        sortby.append(f"avg_{var_}")
        sortby.append(f"std_{var_}")
    trials = trials.sort_values(by=sortby, ascending=ascending)
    return trials


def find_best_trial(trials, value_vars, subdict=None):
    # Sort the trials by the value_vars

    trials = sort_by_value_vars(trials, value_vars)
    # Get the best trial
    best_trial = trials.iloc[1]
    values = best_trial[value_vars]
    for value_var in value_vars:
        avg = f"avg_{value_var}"
        std = f"std_{value_var}"
        print(f"avg_{value_var}: {best_trial[avg]}")
        print(f"std_{value_var}: {best_trial[std]}")
        values[f"avg_{value_var}"] = best_trial[avg]
        del best_trial[avg]
        values[f"std_{value_var}"] = best_trial[std]
        del best_trial[std]
        if value_var in best_trial:
            del best_trial[value_var]
    return best_trial, values


def remove_hydra_syntax(trial):
    # Remove hydra syntax from the trial
    trial = trial.rename(lambda x: x.replace("+", ""))
    trial = trial.rename(lambda x: "++" + x if "." in x else x)
    return trial


def merge_best_with_default(best_trial, default_config):
    best_trial = best_trial.to_dict()
    # best_trial = {f"++{k}": v for k, v in best_trial.items()}
    best_trial = [f"{k}={v}" for k, v in best_trial.items()]
    # Merge the best trial with the default config
    folder = Path(default_config).parent.as_posix()
    file = Path(default_config).name
    logger.info(f"Loading config: {file} from {folder}")
    with initialize(config_path=folder):
        cfg = compose(config_name=file, overrides=best_trial)
    return cfg


def save_best_trial(best_trial, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    best_trial = OmegaConf.to_container(best_trial)
    # Make sure that the directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(best_trial, f)


def find_subset(data, subset):
    # assume that subset is a list of strings, where var=val
    # split on '=' and find the subset
    subset = {k: v for k, v in [s.split("=") for s in subset]}
    for k, v in subset.items():
        data = data[data[k] == v]
    return data


def main(
    study_name,
    storage_path,
    default_path,
    config_path,
    output_path,
    subset=[],
    exclude=[],
):
    # Query the optuna database
    trials, id_vars, value_vars = query_optuna_db(
        study_name=study_name,
        storage_path=storage_path,
    )
    if len(exclude) > 0:
        id_vars = [col for col in id_vars if col not in exclude]
    trials = find_mean_std(trials, id_vars, value_vars)
    default_path = Path(default_path)
    # output_path is the default path with the stem replaced by best_${study_name}.yaml
    file_name = default_path.name
    subconf = default_path.parent.name
    conf = default_path.parent.parent.name
    if conf == ".":
        conf = subconf
        subconf = None
    if len(subset) > 0:
        trials = find_subset(trials, subset)
    best_trial, _ = find_best_trial(trials, value_vars, subdict=subconf)
    best_trial = remove_hydra_syntax(best_trial)
    merged = merge_best_with_default(best_trial, config_path)
    print(f"Configuration: {conf}")
    print(f"Subconfiguration: {subconf}")
    print(f"File: {file_name}")
    print(f"Output Path: {output_path}")
    print(f"ID variables: {id_vars}")
    # Save the best trial to the output path
    save_best_trial(merged, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the best trial for a given study.",
    )
    parser.add_argument(
        "-n",
        "--study_name",
        type=str,
        required=True,
        help="Optuna study name",
    )
    parser.add_argument(
        "-p",
        "--storage_path",
        type=str,
        required=True,
        help="Optuna storage path",
    )
    parser.add_argument(
        "-d",
        "--default_path",
        type=str,
        required=True,
        help="Default path",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=False,
        help="Configuration path",
    )
    parser.add_argument(
        "-s",
        "--subset",
        nargs="+",
        required=False,
        help="Subset of the data to consider",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Output path",
    )
    parser.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        required=False,
        help="Exclude these columns",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(
        study_name=args.study_name,
        storage_path=args.storage_path,
        default_path=args.default_path,
        config_path=args.config_path,
        output_path=args.output_path,
        subset=args.subset,
        exclude=args.exclude,
    )
