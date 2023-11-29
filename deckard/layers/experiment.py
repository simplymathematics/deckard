import logging
from pathlib import Path
import dvc.api
from hydra.utils import instantiate

import yaml
import argparse
from copy import deepcopy
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
        f"Getting params for stage {stage} from {params_file} and {pipeline_file} in {Path(directory).resolve().as_posix()}.",
    )
    stage = [stage] if not isinstance(stage, list) else stage
    params = dvc.api.params_show(stages=stage)
    params.update({"_target_": "deckard.base.experiment.Experiment"})
    files = dvc.api.params_show(pipeline_file, stages=stage, repo=directory)
    unflattened_files = unflatten_dict(files)
    params["files"] = dict(unflattened_files.get("files", unflattened_files))
    params["files"]["_target_"] = "deckard.base.files.FileConfig"
    params["files"]["stage"] = stage
    params["stage"] = stage
    if name is not None:
        params["files"]["name"] = name
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
    files = deepcopy(exp.files())
    params_file = Path(files["score_dict_file"]).with_name("params.yaml").as_posix()
    Path(params_file).parent.mkdir(exist_ok=True, parents=True)
    with Path(params_file).open("w") as f:
        yaml.dump(params, f)
    score = exp()
    return id_, score


def get_stages(pipeline_file="dvc.yaml", stages=None, repo=None):
    with Path(repo, pipeline_file).open("r") as f:
        pipeline = yaml.safe_load(f)["stages"]
    def_stages = list(pipeline.keys())
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


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    dvc_parser = argparse.ArgumentParser()
    dvc_parser.add_argument("stage", type=str, nargs="*", default=None)
    dvc_parser.add_argument("--verbosity", type=str, default="INFO")
    dvc_parser.add_argument("--params_file", type=str, default="params.yaml")
    dvc_parser.add_argument("--pipeline_file", type=str, default="dvc.yaml")
    dvc_parser.add_argument("--config_dir", type=str, default=None)
    dvc_parser.add_argument("--config_file", type=str, default="default")
    dvc_parser.add_argument("--dvc_repository", type=str, default=None)
    args = dvc_parser.parse_args()
    if args.config_dir is not None:
        args.config_dir = Path(args.config_dir).resolve().as_posix()
        assert args.config_file is not None, "Please specify a config file."
    if (
        args.config_dir is not None
        and Path(args.config_dir, args.config_file).is_file()
    ):
        save_params_file(
            config_dir=args.config_dir,
            config_file=args.config_file,
            params_file=args.params_file,
        )
    else:
        pass
    logging.basicConfig(
        level=args.verbosity,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    dvc_repository = args.dvc_repository if args.dvc_repository is not None else "."
    results = run_stages(
        stages=args.stage,
        pipeline_file=args.pipeline_file,
        params_file=args.params_file,
        repo=dvc_repository,
    )
