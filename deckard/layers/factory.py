import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import dvc.api
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import argparse

from ..base.utils import flatten_dict, my_hash, unflatten_dict

logger = logging.getLogger(__name__)

__all__ = ["ExperimentFactory"]


@dataclass
class ExperimentFactory:
    config_dir: str = "conf"
    config_file: str = "default.yaml"
    working_dir: str = "."
    params_file: str = "params.yaml"
    pipeline_file: str = "dvc.yaml"
    output_dir: str = "output"
    stage: str = None
    kwargs: dict = None

    def __init__(
        self,
        config_dir: str = "conf",
        working_dir: str = ".",
        params_file: str = "params.yaml",
        pipeline_file: str = "dvc.yaml",
        stage=None,
        config_file="default.yaml",
        output_dir="output",
        **kwargs,
    ):
        self.config_dir = config_dir
        self.config_file = config_file
        self.working_dir = str(Path(working_dir).as_posix())
        self.params_file = params_file
        self.pipeline_file = pipeline_file
        self.stage = stage

        self.output_dir = output_dir
        self.params = self.get_params()
        self.exp = instantiate(self.params)
        self.name = self.exp.name

    def __hash__(self):
        return my_hash(self)

    def _find_stage_params(self, **kwargs):
        stage = self.stage
        working_dir = Path(self.working_dir).resolve()
        old_params = dvc.api.params_show(
            self.params_file, stages=[stage], repo=working_dir,
        )
        old_params = flatten_dict(old_params)
        new_params = flatten_dict(kwargs)
        new_trunc_keys = [".".join(key.split(".")[:-1]) for key in new_params.keys()]
        params = {}
        for key in old_params:
            trunc = ".".join(key.split(".")[:-1]) if len(key.split(".")) > 1 else key
            if trunc in new_trunc_keys and key in new_params:
                params[key] = new_params[key]
            else:
                params[key] = old_params[key]
        # Setup the files
        params = unflatten_dict(params)
        params["files"] = {}
        files = dvc.api.params_show(
            self.pipeline_file, stages=[stage], repo=self.working_dir,
        )
        unflattened_files = unflatten_dict(files).pop("files", {})
        params["files"].update(**unflattened_files)
        params["files"].update({"_target_": "deckard.base.files.FileConfig"})
        return flatten_dict(params)

    def get_params(self):
        config_dir = str(Path(self.working_dir, self.config_dir).as_posix())
        with initialize_config_dir(
            config_dir=config_dir, version_base=None, job_name="experiment_factory",
        ):
            cfg_old = compose(
                config_name=str(
                    Path(self.working_dir, self.config_dir, self.config_file),
                ),
            )
            cfg_old = deepcopy(OmegaConf.to_container(cfg_old, resolve=True))
            flattened = flatten_dict(cfg_old)
            if self.stage is not None:
                flattened = self._find_stage_params(**flattened)
                cfg = unflatten_dict(flattened)
            else:
                cfg = cfg_old
        if "files" in cfg:
            pass
        else:
            cfg["files"] = {}
        cfg["files"]["_target_"] = "deckard.base.files.FileConfig"
        # cfg['files'].update({"_target_": "deckard.base.files.FileConfig"})
        if "attack_file" in cfg["files"] and cfg["files"]["attack_file"] is not None:
            cfg["files"]["attack_file"] = str(
                Path(cfg["files"]["attack_file"])
                .with_name(my_hash(cfg["attack"]))
                .as_posix(),
            )
        if "model_file" in cfg["files"] and cfg["files"]["model_file"] is not None:
            cfg["files"]["model_file"] = str(
                Path(cfg["files"]["model_file"])
                .with_name(my_hash(cfg["model"]))
                .as_posix(),
            )
        if "data_file" in cfg["files"] and cfg["files"]["data_file"] is not None:
            cfg["files"]["data_file"] = str(
                Path(cfg["files"]["data_file"])
                .with_name(my_hash(cfg["data"]))
                .as_posix(),
            )
        cfg["stage"] = self.stage
        cfg["files"]["stage"] = self.stage
        cfg["files"]["name"] = my_hash(cfg)
        cfg["name"] = my_hash(cfg)
        cfg.update({"_target_": "deckard.base.experiment.Experiment"})
        return cfg

    # def save_stage(self):
    #     params = self.params
    #     in_pipe =  self.pipeline_file
    #     if self.stage is None:
    #         path = Path("queue", self.name)
    #     else:
    #         path = Path(self.stage, self.name)
    #     working_dir = Path(self.working_dir, self.output_dir).resolve()
    #     sub_dir = Path(self.working_dir, self.output_dir, path).resolve()
    #     path = str(Path(os.path.relpath( working_dir, sub_dir)).as_posix())
    #     out_pipe = path / self.pipeline_file
    #     out_params = path / self.params_file
    #     Path(path).mkdir(parents=True, exist_ok=True)
    #     if self.stage is None:
    #         with open(in_pipe, "r") as f:
    #             pipeline = yaml.safe_load(f)['stages']
    #         pipeline = {f"{stage}_{self.name}": pipeline[stage] for stage in pipeline}
    #     else:
    #         with open(in_pipe, "r") as f:
    #             pipeline = yaml.safe_load(f)['stages'][self.stage]
    #         pipeline['cmd'] = pipeline['cmd'].replace(self.stage, f"{self.stage}_{self.name}")
    #         pipeline = {f"{self.stage}_{self.name}": pipeline}
    #     pipeline = {"stages": pipeline}
    #     with open(out_pipe, "w") as f:
    #         yaml.safe_dump(pipeline, f)
    #     assert Path(out_pipe).exists()
    #     with open(out_params, "w") as f:
    #         yaml.safe_dump(params, f)
    #     assert Path(out_params).exists()

    def save_stage_params(self):
        files = deepcopy(self.exp.files)()
        folder = Path(files["score_dict_file"]).parent
        path = Path(folder, Path(self.params_file).name)
        with open(path, "w") as f:
            yaml.safe_dump(self.params, f)
        assert Path(path).exists()

    def __call__(self):
        params = self.get_params()
        exp = instantiate(params)
        score = exp()
        self.save_stage_params()
        return score


@dataclass
class OldExperimentFactory:
    config_dir: str = "conf"
    config_file: str = "default.yaml"
    working_dir: str = "."
    params_file: str = "params.yaml"
    pipeline_file: str = "dvc.yaml"
    output_dir: str = "output"
    stage: str = None
    kwargs: dict = None

    def __init__(
        self,
        config_dir: str = "conf",
        working_dir: str = ".",
        params_file: str = "params.yaml",
        pipeline_file: str = "dvc.yaml",
        stage=None,
        config_file="default.yaml",
        output_dir="output",
        **kwargs,
    ):
        self.config_dir = config_dir
        self.config_file = config_file
        self.working_dir = str(Path(working_dir).as_posix())
        self.params_file = params_file
        self.pipeline_file = pipeline_file
        self.stage = stage
        self.kwargs = kwargs if kwargs is not None else {}
        self.params = self.get_params()
        self.exp = instantiate(self.params)
        self.output_dir = output_dir
        self.name = self.exp.name

    def __hash__(self):
        return my_hash(self)

    def _find_stage_params(self, **kwargs):
        stage = self.stage
        old_params = dvc.api.params_show(
            self.params_file, stages=[stage], repo=self.working_dir,
        )
        old_params = flatten_dict(old_params)
        new_params = flatten_dict(kwargs)
        new_trunc_keys = [".".join(key.split(".")[:-1]) for key in new_params.keys()]
        params = {}
        for key in old_params:
            trunc = ".".join(key.split(".")[:-1]) if len(key.split(".")) > 1 else key
            if trunc in new_trunc_keys and key in new_params:
                logger.debug(f"Overwriting {key} with {new_params[key]}")
                params[key] = new_params[key]
            else:
                pass
        # Setup the files
        params = unflatten_dict(params)
        params["files"] = {}
        files = dvc.api.params_show(
            self.pipeline_file, stages=[stage], repo=self.working_dir,
        )
        unflattened_files = unflatten_dict(files).pop("files", {})
        params["files"].update(**unflattened_files)
        params["files"].update({"_target_": "deckard.base.files.FileConfig"})
        return flatten_dict(params)

    def get_params(self):
        config_dir = str(Path(self.working_dir, self.config_dir).as_posix())
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg_old = compose(
                config_name=str(
                    Path(self.working_dir, self.config_dir, self.config_file),
                ),
            )
            cfg_old = deepcopy(OmegaConf.to_container(cfg_old, resolve=True))
            flattened = flatten_dict(cfg_old)
            if self.stage is not None:
                flattened = self._find_stage_params(**flattened)
            overrides = self._override(self.kwargs, flattened)
            if len(overrides) > 0:
                cfg_new = compose(config_name=self.config_file, overrides=overrides)
                cfg = OmegaConf.to_container(cfg_new, resolve=True)
            else:
                cfg = deepcopy(cfg_old)
            # TODO: parse stage
            if "files" in cfg:
                pass
            else:
                cfg["files"] = {}
            # cfg['files'].update({"_target_": "deckard.base.files.FileConfig"})
            if (
                "attack_file" in cfg["files"]
                and cfg["files"]["attack_file"] is not None
            ):
                cfg["files"]["attack_file"] = str(
                    Path(cfg["files"]["attack_file"])
                    .with_name(my_hash(cfg["attack"]))
                    .as_posix(),
                )
            if "model_file" in cfg["files"] and cfg["files"]["model_file"] is not None:
                cfg["files"]["model_file"] = str(
                    Path(cfg["files"]["model_file"])
                    .with_name(my_hash(cfg["model"]))
                    .as_posix(),
                )
            if "data_file" in cfg["files"] and cfg["files"]["data_file"] is not None:
                cfg["files"]["data_file"] = str(
                    Path(cfg["files"]["data_file"])
                    .with_name(my_hash(cfg["data"]))
                    .as_posix(),
                )
            if "reports" in cfg and cfg["reports"] is not None:
                if self.stage is None:
                    cfg["reports"] = str(Path(cfg["reports"]), self.name)
                else:
                    cfg["reports"] = str(Path(cfg["reports"]), self.stage, self.name)
            cfg.update({"_target_": "deckard.base.experiment.Experiment"})
        return cfg

    def _override(self, override_params, defaults, prefix=""):
        overrides = []
        for name in override_params:
            param = override_params[name]
            name = prefix + name
            if isinstance(param, dict):
                overrides += self._override(param, defaults, prefix=name + ".")
            else:
                if param in defaults:
                    logger.info(f"Overwriting {name} with {param}")
                    overrides.append(f"{name}={param}")
                else:
                    logger.info(f"Adding {name} with {param}")
                    overrides.append(f"++{name}={param}")
        return overrides

    def __call__(self):
        params = self.get_params()
        exp = instantiate(params)
        score = exp()
        self.save_stage_params()
        return score


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    dvc_parser = argparse.ArgumentParser()
    dvc_parser.add_argument("--params_file", type=str, default="params.yaml")
    dvc_parser.add_argument("--pipeline_file", type=str, default="dvc.yaml")
    dvc_parser.add_argument("--config_dir", type=str, default="conf")
    dvc_parser.add_argument("--config_file", type=str, default="default.yaml")
    dvc_parser.add_argument("--verbosity", type=str, default="INFO")
    dvc_parser.add_argument(
        "stage", type=str, nargs="?", default=None, help="Stage to run",
    )
    args = dvc_parser.parse_args()
    os.chdir(Path())
    logging.basicConfig(level=args.verbosity)
    if args.stage is None:
        raise ValueError("Please specify a stage to run.")
    else:
        stages = args.stage if isinstance(args.stage, list) else [args.stage]
    config_dir = Path(Path(), args.config_dir).resolve().as_posix()
    total_length = 0
    total_length = len(stages)
    for stage in stages:
        if stage.startswith("+stage="):
            stage = stage.split("=")[-1]
        logger.info(f"Running stage {stage} of {len(stages)}")
        factory = ExperimentFactory(
            config_dir=config_dir,
            config_file=args.config_file,
            params_file=args.params_file,
            pipeline_file=args.pipeline_file,
            stage=stage,
        )
        results = factory()
        logger.info(f"Stage {stage} complete.")
