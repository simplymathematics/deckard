import optuna
import logging
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
import yaml
from ..base.utils import flatten_dict, unflatten_dict

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
    params = flatten_dict(study.best_params)
    best_params = unflatten_dict(params)
    best_params = flatten_dict(best_params)
    overrides = []
    for key, value in best_params.items():
        logger.info(f"Overriding {key} with {value}")
        if not key.startswith("+"):
            overrides.append(f"++{key}={value}")
    with initialize_config_dir(config_dir=config_folder, version_base="1.3"):
        cfg = compose(config_name=default_config, overrides=overrides)
        cfg = OmegaConf.to_container(cfg, resolve=False)
    if params_file is not None:
        if params_file is True:
            if config_subdir is not None:
                params_file = Path(config_folder, f"{config_subdir}", f"{default_config}.yaml")
                params = cfg.get(config_subdir)
            else:
                params_file = Path(config_folder, f"{default_config}.yaml")
                params = cfg
        else:
            if config_subdir is not None:
                params_file = Path(config_folder, f"{config_subdir}", f"{params_file}.yaml")
                params = cfg.get(config_subdir)
            else:
                params = cfg
                params_file = Path(config_folder, f"{params_file}.yaml")
        params_file.parent.mkdir(parents=True, exist_ok=True)
        with open(params_file.with_suffix(".yaml"), "w") as f:
            yaml.dump(params, f)
        assert params_file.exists(), f"{params_file.resolve().as_posix()} does not exist."
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
        with open(Path(args.config_folder, args.default_config).with_suffix(".yaml"), "r") as f:
            default_params = yaml.load(f, Loader=yaml.FullLoader)
        if "hydra" in default_params:
            hydra_params = default_params.pop("hydra")

        study_name = args.study_name
        storage_name = hydra_params["sweeper"]["storage"]
        direction = default_params.get("direction", "maximize")
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
