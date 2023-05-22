import optuna
import logging
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
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
    config_name="default.yaml",
    direction=None,
):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction=direction,
    )
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    if study_csv is not None:
        df.to_csv(study_csv)
    best_params = flatten_dict(study.best_params)
    overrides = []
    for key, value in best_params.items():
        logger.info(f"Overriding {key} with {value}")
        overrides.append(f"++{key}={value}")
    with initialize_config_dir(config_dir=config_folder):
        cfg = compose(config_name=config_name, overrides=overrides)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["_target_"] = "deckard.base.experiment.Experiment"
        flattened = flatten_dict(cfg)
    if params_file is not None:
        if params_file is True:
            if study_name in cfg:
                params = cfg.get(study_name, cfg)
                params_file = Path(config_folder, study_name, "best.yaml")
            elif study_name in flattened:
                params = flattened.get(study_name, flattened)
                params_file = Path(config_folder, f"best_{study_name}.yaml")
            else:
                params_file = Path(config_folder, f"best_{study_name}.yaml")
        with open(params_file, "w") as f:
            yaml.dump(params, f)
    return params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default=True)
    parser.add_argument("--study_type", type=str, default="optuna")
    parser.add_argument("--study_csv", type=str, default=None)
    parser.add_argument("--config_folder", type=str, default=Path(Path(), "conf"))
    parser.add_argument("config_name", type=str)
    parser.add_argument("--verbosity", type=str, default="INFO")
    args = parser.parse_args()

    args.config_folder = Path(args.config_folder).resolve().as_posix()

    if args.study_type == "optuna":
        with open(Path(args.config_folder, args.config_name), "r") as f:
            default_params = yaml.load(f, Loader=yaml.FullLoader)
        if "hydra" in default_params:
            hydra_params = default_params.pop("hydra")

        study_name = hydra_params["sweeper"]["study_name"]
        storage_name = hydra_params["sweeper"]["storage"]
        direction = hydra_params["sweeper"]["direction"]
        find_optuna_best(
            study_name=study_name,
            storage_name=storage_name,
            study_csv=args.study_csv,
            params_file=args.params_file,
            config_folder=args.config_folder,
            config_name=args.config_name,
            direction=direction,
        )
    else:
        raise NotImplementedError(f"Study type {args.study_type} not implemented.")
