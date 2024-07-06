import optuna
import logging
import pandas as pd
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import argparse
import yaml
from ..base.utils import flatten_dict

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def find_optuna_best(
    study_name,
    storage_name,
    study_csv=None,
    params_file=None,
    config_folder=Path(Path(), "conf"),
    default_config="default.yaml",
    config_subdir=None,
    direction="maximize",
):
    logger.info(f"Study name: {study_name}")
    logger.info(f"Storage name: {storage_name}")
    if isinstance(direction, str):
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction=direction,
        )
        directions = [direction]
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            directions=direction,
        )
        directions = direction
    assert isinstance(directions, list), f"Directions is not a list: {type(directions)}"
    df = study.trials_dataframe(attrs=("number", "value", "params"))
    # Find the average of each value over the columns in average_over
    not_these = ["number", "value"]
    val_cols = [
        col
        for col in df.columns
        if col.startswith("values_") and col.split("values_")[-1] not in not_these
    ]
    not_these.extend(val_cols)
    print(f"Not these: {not_these}")
    groupby_cols = [
        col for col in df.columns if col.split("params_")[-1] not in not_these
    ]
    print(f"Groupby cols: {groupby_cols}")
    dfs = df.groupby(groupby_cols)
    new_df = pd.DataFrame(columns=groupby_cols + ["mean", "std", "ntrials", "nuniques"])
    means = []
    stds = []
    ntrials = []
    nuniques = []
    for _, df in dfs:
        # find mean of value
        mean_ = df.value.mean()
        # find the std
        std_ = df.value.std()
        # find the number of trials
        n = df.value.count()
        # find the number of unique trials
        nunique = df.value.nunique()
        means.append(mean_)
        stds.append(std_)
        ntrials.append(n)
        nuniques.append(nunique)
        # add first row of df to new_df
        new_df = pd.concat([new_df, df.head(1)])
    new_df.drop(columns=["value"], inplace=True)
    new_df["mean"] = means
    new_df["std"] = stds
    new_df["ntrials"] = ntrials
    new_df["nuniques"] = nuniques
    for direction in directions:
        assert direction in [
            "minimize",
            "maximize",
        ], f"Direction {direction} not recognized."
    directions = [False if x == "maximize" else True for x in directions]
    assert isinstance(new_df, pd.DataFrame), f"df is not a dataframe: {type(df)}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction=direction,
    )

    if study_csv is not None:
        save_study(study_csv, study)
    # To dotlist
    best_params = flatten_dict(study.best_params)
    more_params = flatten_dict(study.best_trial.user_attrs)
    even_more_params = flatten_dict(study.best_trial.system_attrs)
    logger.debug(f"Best params: {best_params}")
    logger.debug(f"Best user params: {more_params}")
    logger.debug(f"Best system params: {even_more_params}")
    # Merge all the params
    best_params = OmegaConf.to_container(
        OmegaConf.merge(best_params, more_params, even_more_params),
        resolve=False,
    )
    # to dotlist
    best_params = flatten_dict(best_params)
    overrides = []
    # Changing the keys to hydra override format
    for key, value in best_params.items():
        if (
            key.startswith("++") or key.startswith("~") or key.startswith("--")
        ):  # reserved meaning
            pass
        elif key.startswith("+"):  # appends to config
            key = "++" + key[1:]  # force override
        elif key.startswith("~"):  # appends to config
            pass
        else:
            key = "++" + key  # force override
        if config_subdir is None:
            overrides.append(f"{key}={value}")
        else:  # if we are using a subdir, we need to remove the directory from the key
            if (
                key.startswith(f"++{config_subdir}.")
                or key.startswith(f"~{config_subdir}.")
                or key.startswith(f"--{config_subdir}.")
            ):
                key = key.replace(f"{config_subdir}.", "")
                overrides.append(f"{key}={value}")
                logger.info(f"Adding {key} to param list")
            else:
                logger.debug(f"Skipping {key} because it is not in {config_subdir}")
    return overrides


def find_best_params(study):
    best_params = flatten_dict(study.best_params)
    more_params = flatten_dict(study.best_trial.user_attrs)
    even_more_params = flatten_dict(study.best_trial.system_attrs)
    logger.debug(f"Best params: {best_params}")
    logger.debug(f"Best user params: {more_params}")
    logger.debug(f"Best system params: {even_more_params}")
    # Merge all the params
    best_params = OmegaConf.to_container(
        OmegaConf.merge(best_params, more_params, even_more_params),
        resolve=False,
    )
    # to dotlist
    best_params = flatten_dict(best_params)
    return best_params



def prepare_overrides(config_subdir, best_params):
    overrides = []
    # Changing the keys to hydra override format
    for key, value in best_params.items():
        if (
            key.startswith("++") or key.startswith("~") or key.startswith("--")
        ):  # reserved meaning
            pass
        elif key.startswith("+"):  # appends to config
            key = "++" + key[1:]  # force override
        elif key.startswith("~"):  # appends to config
            pass
        else:
            key = "++" + key  # force override
        if config_subdir is None:
            overrides.append(f"{key}={value}")
        else:  # if we are using a subdir, we need to remove the directory from the key
            if (
                key.startswith(f"++{config_subdir}.")
                or key.startswith(f"~{config_subdir}.")
                or key.startswith(f"--{config_subdir}.")
            ):
                key = key.replace(f"{config_subdir}.", "")
                overrides.append(f"{key}={value}")
                logger.info(f"Adding {key} to param list")
            else:
                logger.debug(f"Skipping {key} because it is not in {config_subdir}")
    return overrides


def find_best_params(study):
    best_params = flatten_dict(study.best_params)
    more_params = flatten_dict(study.best_trial.user_attrs)
    even_more_params = flatten_dict(study.best_trial.system_attrs)
    logger.debug(f"Best params: {best_params}")
    logger.debug(f"Best user params: {more_params}")
    logger.debug(f"Best system params: {even_more_params}")
    # Merge all the params
    best_params = OmegaConf.to_container(
        OmegaConf.merge(best_params, more_params, even_more_params),
        resolve=False,
    )
    # to dotlist
    best_params = flatten_dict(best_params)
    return best_params


def save_study(study_csv, study):
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    Path(study_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(study_csv)


def create_new_config_in_subdir(
    params_file,
    config_folder,
    default_config,
    config_subdir,
    params,
):
    if params_file is True:
        if config_subdir is not None:
            params_file = Path(
                config_folder,
                f"{config_subdir}",
                f"{default_config}.yaml",
            )
        else:
            params_file = Path(config_folder, f"{default_config}.yaml")
    else:
        if config_subdir is not None:
            params_file = Path(
                config_folder,
                f"{config_subdir}",
                f"{params_file}.yaml",
            )
        else:
            params_file = Path(config_folder, f"{params_file}.yaml")
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file.with_suffix(".yaml"), "w") as f:
        yaml.dump(params, f)
    assert params_file.exists(), f"{params_file.resolve().as_posix()} does not exist."

    return params_file


def override_config(
    config_folder,
    default_config,
    overrides,
    config_subdir=None,
):
    if config_subdir is not None:
        config_folder = Path(config_folder, config_subdir)
        config_folder = config_folder.resolve().as_posix()
    with initialize_config_dir(config_dir=config_folder, version_base="1.3"):
        cfg = compose(config_name=default_config, overrides=overrides)
    cfg = OmegaConf.to_container(cfg, resolve=False)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--params_file", type=str, default=True)

parser.add_argument("--study_csv", type=str, default=None)
parser.add_argument("--config_folder", type=str, default=Path(Path(), "conf"))
parser.add_argument("--default_config", type=str, default="default")
parser.add_argument("--config_subdir", type=str, default=None)
parser.add_argument("--study_name", type=str, required=True)
parser.add_argument("--config_name", type=str)
parser.add_argument("--verbosity", type=str, default="INFO")
parser.add_argument("--storage_name", type=str, required=True)
parser.add_argument("--direction", type=str, default="maximize")
parser.add_argument("--study_type", type=str, default="optuna")


def main(find_optuna_best, args):
    args.config_folder = Path(args.config_folder).resolve().as_posix()
    logging
    if args.study_type == "optuna":
        direction = args.direction
        if len(direction) == 1:
            direction = direction[0]
        find_optuna_best(
            study_name=args.study_name,
            storage_name=args.storage_name,
            study_csv=args.study_csv,
            params_file=args.params_file,
            config_folder=args.config_folder,
            config_subdir=args.config_subdir,
            default_config=args.default_config,
            direction=direction,
        )
    else:
        raise NotImplementedError(f"Study type {args.study_type} not implemented.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(find_optuna_best, args)
