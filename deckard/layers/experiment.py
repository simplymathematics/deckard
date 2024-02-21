#! /usr/bin/env python

import logging
from pathlib import Path
import dvc.api
from hydra.utils import instantiate
from dulwich.errors import NotGitRepository
import yaml
import argparse
from copy import deepcopy
from omegaconf import OmegaConf
from ..base.utils import unflatten_dict
from .utils import save_params_file

logger = logging.getLogger(__name__)

__all__ = [
    "get_dvc_stage_params",
    "run_stage",
    "get_stages",
    "run_stages",
]


def get_dvc_stage_params(
    stage,
    params_file="params.yaml",
    pipeline_file="dvc.yaml",
    directory=".",
    name=None,
):
    logger.info(
        f"Getting params for stage {stage} from {params_file} and {pipeline_file} in {directory}.",
    )
    params = dvc.api.params_show(stages=stage)
    params.update({"_target_": "deckard.base.experiment.Experiment"})
    pipe_params = dvc.api.params_show(pipeline_file, stages=stage, repo=directory)
    pipe_params = unflatten_dict(pipe_params)
    params["files"] = dict(pipe_params.pop("files", pipe_params))
    params["files"]["_target_"] = "deckard.base.files.FileConfig"
    params["files"]["stage"] = stage
    params["stage"] = stage
    if name is not None:
        params["files"]["name"] = name
    # Merge remaining params
    params = OmegaConf.merge(params, pipe_params)
    return params





def get_stages(pipeline_file="dvc.yaml", stages=None, repo=None):
    try:
        def_stages = list(
            dvc.api.params_show(pipeline_file, repo=repo)["stages"].keys(),
        )
    except NotGitRepository:
        raise ValueError(
            f"Directory {repo} is not a dvc repository. Please run `dvc init` in {repo} and try again.",
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

def run_stage(
    params_file="params.yaml",
    pipeline_file="dvc.yaml",
    directory=".",
    stage=None,
    overrides=None,
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
    old_params = deepcopy(params)   
    if overrides is not None:
        # convert from dot notation to nested dict
        overrides = OmegaConf.from_dotlist(overrides)
        params = OmegaConf.merge(params, overrides)
        params = OmegaConf.to_container(params, resolve=True)
        assert params != old_params, f"Params are the same as before overrides: {overrides}"
    params = OmegaConf.create(params)
    params = OmegaConf.to_container(params, resolve=True)
    exp = instantiate(params)
    id_ = exp.name
    files = deepcopy(exp.files())
    params_file = Path(files["score_dict_file"]).with_name("params.yaml").as_posix()
    Path(params_file).parent.mkdir(exist_ok=True, parents=True)
    with Path(params_file).open("w") as f:
        yaml.dump(params, f)
    score = exp()
    return id_, score

def run_stages(stages, pipeline_file="dvc.yaml", params_file="params.yaml", repo=None,):
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
    dvc_parser.add_argument("--overrides", nargs="*", default=[], type=str)
    args = dvc_parser.parse_args()
    logging.basicConfig(
        level=args.verbosity,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    config_dir = Path(args.workdir, args.config_dir).relative_to(Path()).as_posix()
    if (args.overrides is not None and len(args.overrides) > 0):
        save_params_file(
            config_dir=config_dir,
            config_file=args.config_file,
            params_file=args.params_file,
            overrides=args.overrides,
        )

    else:
        logger.info(f"Using existing params file {args.params_file} in directory {args.workdir}")
    results = run_stages(
        stages=args.stage,
        pipeline_file=args.pipeline_file,
        params_file=args.params_file,
        repo=args.workdir,
    )