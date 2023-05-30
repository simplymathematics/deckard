import logging
from pathlib import Path
import dvc.api
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from dulwich.errors import NotGitRepository
import yaml
import argparse
from ..base.utils import unflatten_dict

logger = logging.getLogger(__name__)

__all__ = [
    "get_dvc_stage_params",
    "run_stage",
    "get_stages",
    "run_stages",
    "save_params_file",
]


def get_dvc_stage_params(
    stage,
    params_file="params.yaml",
    pipeline_file="dvc.yaml",
    directory=".",
):
    logger.info(
        f"Getting params for stage {stage} from {params_file} and {pipeline_file} in {directory}.",
    )
    params = dvc.api.params_show(params_file, stages=stage, repo=directory)
    params.update({"_target_": "deckard.base.experiment.Experiment"})
    files = dvc.api.params_show(pipeline_file, stages=stage, repo=directory)
    unflattened_files = unflatten_dict(files)
    params["files"] = dict(unflattened_files.get("files", unflattened_files))
    params["files"]["_target_"] = "deckard.base.files.FileConfig"
    params["files"]["stage"] = stage
    return params


def run_stage(
    params_file="params.yaml",
    pipeline_file="dvc.yaml",
    directory=".",
    stage=None,
):
    logger.info(
        f"Running stage {stage} with params_file: {params_file} and pipeline_file: {pipeline_file} in directory {directory}",
    )
    params = get_dvc_stage_params(
        stage=stage,
        params_file=params_file,
        pipeline_file=pipeline_file,
        directory=directory,
    )
    exp = instantiate(params)
    id_ = exp.name
    score = exp()
    return id_, score


def get_stages(pipeline_file="dvc.yaml", stages=None, repo=None):
    try:
        def_stages = list(
            dvc.api.params_show(pipeline_file, repo=repo)["stages"].keys(),
        )
    except NotGitRepository:
        raise ValueError(
            f"Directory {repo} is not a git repository. Please run `dvc init` in {repo} and try again.",
        )
    if stages is None or stages == []:
        raise ValueError(f"Please specify one or more stage(s) from {def_stages}")
    elif isinstance(stages, str):
        stages = [stages]
    else:
        assert isinstance(stages, list), f"args.stage is of type {type(stages)}"
        for stage in stages:
            assert (
                stage in def_stages
            ), f"Stage {stage} not found in {pipeline_file}. Available stages: {def_stages}"
    return stages


def run_stages(stages, pipeline_file="dvc.yaml", params_file="params.yaml", repo=None):
    results = {}
    stages = get_stages(stages=stages, pipeline_file=pipeline_file, repo=repo)
    for stage in stages:
        id_, score = run_stage(
            stage=stage,
            pipeline_file=pipeline_file,
            params_file=params_file,
            directory=repo,
        )
        results[id_] = score
    return results


def save_params_file(
    config_dir="conf",
    config_file="default",
    params_file="params.yaml",
):
    config_dir = str(Path(Path(), config_dir).absolute().as_posix())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_file)
        params = OmegaConf.to_container(cfg, resolve=True)
        with open(params_file, "w") as f:
            yaml.dump(params, f)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    dvc_parser = argparse.ArgumentParser()
    dvc_parser.add_argument("stage", type=str, nargs="*", default=None)
    dvc_parser.add_argument("--verbosity", type=str, default="INFO")
    dvc_parser.add_argument("--params_file", type=str, default="params.yaml")
    dvc_parser.add_argument("--pipeline_file", type=str, default="dvc.yaml")
    dvc_parser.add_argument("--config_dir", type=str, default="conf")
    dvc_parser.add_argument("--config_file", type=str, default="default")
    dvc_parser.add_argument("--workdir", type=str, default=".")
    args = dvc_parser.parse_args()
    config_dir = Path(Path(), args.config_dir).resolve().as_posix()
    save_params_file(
        config_dir=config_dir,
        config_file=args.config_file,
        params_file=args.params_file,
    )
    logging.basicConfig(
        level=args.verbosity,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    results = run_stages(
        stages=args.stage,
        pipeline_file=args.pipeline_file,
        params_file=args.params_file,
        repo=args.workdir,
    )