import optuna
import logging
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import pandas as pd
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
    direction=None,
    average_over = ["++data.sample.random_state"],
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
    not_these = ['number', 'value']
    val_cols = [col for col in df.columns if col.startswith("values_") and col.split("values_")[-1] not in not_these]
    not_these.extend(val_cols)
    not_these.extend(average_over)
    print(f"Not these: {not_these}")
    groupby_cols = [col for col in df.columns if col.split("params_")[-1] not in not_these]
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
    for col in average_over:
        new_df.drop(columns=[ f"params_{col}"], inplace=True)
    new_df["mean"] = means
    new_df["std"] = stds
    new_df["ntrials"] = ntrials
    new_df["nuniques"] = nuniques
    param_cols = [col for col in new_df.columns if col.startswith("params_")]
    for direction in directions:
        assert direction in ["minimize", "maximize"], f"Direction {direction} not recognized."
    directions = [False if x == "maximize" else True for x in directions]
    assert isinstance(new_df, pd.DataFrame), f"df is not a dataframe: {type(df)}"
    sorted_df = new_df.sort_values(by = 'mean', ascending = directions)
    if study_csv is not None:
        Path(study_csv).parent.mkdir(parents=True, exist_ok=True)
        sorted_df.to_csv(study_csv, index=False)
    params = new_df.iloc[0][param_cols].to_dict()
    best_params = flatten_dict(params)
    logger.debug(f"Best params: {best_params}")
    overrides = []
    for key, value in best_params.items():
        logger.info(f"Overriding {key} with {value}")
        if key.startswith("++"):
            pass
        elif key.startswith("+"):
            key = key.replace("+", "++")
        else:
            key = f"++{key}"
        overrides.append(f"{key}={value}")
    with initialize_config_dir(config_dir=config_folder, version_base="1.3"):
        cfg = compose(config_name=default_config, overrides=overrides)
        cfg = OmegaConf.to_container(cfg, resolve=False)
    if params_file is not None:
        if params_file is True:
            if config_subdir is not None:
                params_file = Path(
                    config_folder,
                    f"{config_subdir}",
                    f"{default_config}.yaml",
                )
                params = cfg.get(config_subdir)
            else:
                params_file = Path(config_folder, f"best_{default_config}.yaml")
                params = cfg
        else:
            if config_subdir is not None:
                params_file = Path(
                    config_folder,
                    f"{config_subdir}",
                    f"{params_file}.yaml",
                )
                params = cfg.get(config_subdir)
            else:
                params = cfg
                params_file = Path(config_folder, f"{params_file}.yaml")
        params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(params_file.with_suffix(".yaml"), "w") as f:
            yaml.dump(params, f)
        assert (
            params_file.exists()
        ), f"{params_file.resolve().as_posix()} does not exist."
    return params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default=True)
    parser.add_argument("--study_type", type=str, default="optuna")
    parser.add_argument("--study_csv", type=str, default=None)
    parser.add_argument("--config_folder", type=str, default=Path(Path(), "conf"))
    parser.add_argument("--default_config", type=str, default="default")
    parser.add_argument("--config_subdir", type=str, default=None)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--verbosity", type=str, default="INFO")
    args = parser.parse_args()

    args.config_folder = Path(args.config_folder).resolve().as_posix()

    if args.study_type == "optuna":
        with initialize_config_dir(config_dir=args.config_folder, version_base="1.3"):
            default_params = compose(config_name=args.default_config, return_hydra_config=True, overrides=["++hydra.job.num=0", "++hydra.job_logging.handlers.file.filename=null"])
        default_params = OmegaConf.to_container(OmegaConf.create(default_params), resolve=True)
        if "hydra" in default_params:
            hydra_params = default_params.pop("hydra")
        else:
            raise ValueError("No hydra params found in default config.")
        study_name = args.study_name
        storage_name = hydra_params["sweeper"]["storage"]
        direction = default_params.get("direction", "maximize")
        logger.info(f"Direction: {direction}")
        find_optuna_best(
            study_name=study_name,
            storage_name=storage_name,
            study_csv=args.study_csv,
            params_file=args.params_file,
            config_folder=args.config_folder,
            config_subdir=args.config_subdir,
            default_config=args.default_config,
            direction=direction,
        )
    else:
        raise NotImplementedError(f"Study type {args.study_type} not implemented.")
