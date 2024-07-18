import logging
from pathlib import Path
import re

from hydra.errors import OverrideParseException
from omegaconf import OmegaConf
from omegaconf import SCMode
from copy import deepcopy
import yaml
from hydra import initialize_config_dir, compose
import dvc.api
from hydra.utils import instantiate
from dulwich.errors import NotGitRepository


from numpy import nan
from ..base.utils import my_hash, flatten_dict, unflatten_dict

logger = logging.getLogger(__name__)

deckard_nones = [
    None,
    "None",
    "",
    "nan",
    "NAN",
    "null",
    "NULL",
    "Null",
    "none",
    "NONE",
    nan,
    "NaN",
]


def find_conf_files(
    config_subdir,
    config_dir,
    config_name=None,
    config_regex=None,
    default_file=None,
):
    if config_name is not None:
        assert config_regex is None, "Cannot specify both config_name and config_regex"
        config_dir = Path(Path(), config_dir).resolve().as_posix()
        sub_dir = Path(config_dir, config_subdir)
        files = [Path(sub_dir, config_name)]
    elif config_regex is not None:
        assert config_name is None, "Cannot specify both config_name and config_regex"
        config_dir = Path(Path(), config_dir).resolve().as_posix()
        sub_dir = Path(config_dir, config_subdir)
        files = sub_dir.glob(config_regex)
    elif default_file is not None:
        assert config_name is None, "Cannot specify both config_name and config_regex"
        config_dir = Path(Path(), config_dir).resolve().as_posix()
        sub_dir = Path(config_dir, config_subdir)
        files = [default_file]
    else:  # pragma: no cover
        raise ValueError(
            "Must specify either config_name or config_regex or default_file",
        )
    files = [file.as_posix() for file in files]
    return files


def get_overrides(overrides=None):
    if overrides is None:
        overrides = {}
    else:
        if isinstance(overrides, str):
            overrides = overrides.split(",")
        if isinstance(overrides, list):
            overrides = {
                entry.split("=")[0]: entry.split("=")[1] for entry in overrides
            }
        if isinstance(overrides, dict):
            new_dict = deepcopy(overrides)
            for k, v in new_dict.items():
                if k.startswith("++"):
                    overrides[k] = v
                elif k.startswith("+"):
                    overrides[f"++{k[1:]}"] = v
                elif k.startswith("~"):
                    overrides[f"~{k[2:]}"] = v
                else:
                    overrides[f"++{k}"] = v

        # assert isinstance(overrides, dict), f"Expected list, got {type(overrides)}"
    # if key is not None and len(overrides) > 0:
    #     overrides.pop(f"{key}.name", None)
    #     overrides.pop(f"files.{key}_file", None)
    #     overrides[f"++{key}.name"] = Path(file).stem
    #     overrides[f"++files.{key}_file"] = Path(file).stem
    #     overrides[f"{key}"] = Path(file).stem
    #     overrides["++stage"] = key
    return overrides


def compose_experiment(file, config_dir, overrides=None, default_file="default.yaml"):
    if hasattr(file, "as_posix"):
        file = file.as_posix()
    if overrides in [None, "", "None", "none", "NONE", "null", "Null", "NULL"]:
        overrides = []
    elif isinstance(overrides, str):
        overrides = overrides.split(",")
    if isinstance(overrides, list):
        pass
    elif isinstance(overrides, dict):
        new_dict = deepcopy(overrides)
        for k, v in new_dict.items():
            if k.startswith("++"):
                overrides[k] = v
            elif k.startswith("+"):
                overrides[f"++{k[1:]}"] = v
            elif k.startswith("--"):
                overrides[f"++{k[2:]}"] = v
            else:
                overrides[f"++{k}"] = v
    else:
        raise TypeError(f"Expected list or dict, got {type(overrides)}")
    assert isinstance(file, str), f"Expected str, got {type(file)}"
    # file = Path(data_conf_dir, file).as_posix()
    logger.info(f"Running experiment in config_dir: {config_dir}")
    logger.info(f"Running experiment with config_name: {file}")
    config_dir = Path(Path(), config_dir).resolve().as_posix()
    OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        try:
            cfg = compose(config_name=Path(default_file).stem, overrides=overrides)
        except OverrideParseException:
            raise ValueError(f"Failed to parse overrides: {overrides}")
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["_target_"] = "deckard.Experiment"
        id_ = str(my_hash(cfg))
        cfg["name"] = id_
        cfg["files"]["name"] = id_
        return cfg


def save_params_file(
    config_dir="conf",
    config_file="default",
    params_file="params.yaml",
    working_directory=".",
    overrides=[],
):
    config_dir = str(Path(working_directory, config_dir).absolute().as_posix())
    logger.info(f"Running save_params_file in config_dir: {config_dir}")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_file, overrides=overrides)
    params = OmegaConf.to_container(
        cfg,
        resolve=True,
        structured_config_mode=SCMode.DICT,
    )
    with open(params_file, "w") as f:
        yaml.dump(params, f)
    logger.info(f"Saved params file to {params_file}")
    assert Path(params_file).exists(), f"Failed to save params file to {params_file}"
    return None


def get_dvc_stage_params(
    stage,
    params_file="params.yaml",
    pipeline_file="dvc.yaml",
    directory=".",
    name=None,
):
    main_stage = stage.split("@")[0]
    sub_stage = stage.split("@")[1] if main_stage != stage else None
    logger.info(
        f"Getting params for stage {main_stage} from {params_file} and {pipeline_file} in {directory}.",
    )
    params = dvc.api.params_show(stages=main_stage, repo=directory)
    params.update({"_target_": "deckard.base.experiment.Experiment"})
    params = OmegaConf.to_container(OmegaConf.create(params), resolve=True)
    flat_params = flatten_dict(params)
    keys = dvc.api.params_show(pipeline_file, stages=stage, repo=directory).keys()
    if "stages" in keys:

        pipe_params = dvc.api.params_show(pipeline_file, stages=stage, repo=directory)[
            "stages"
        ]
        if sub_stage is None:
            pipe_params = pipe_params[stage]
        else:
            pipe_params = pipe_params[main_stage]["do"]
        file_list = []
        for key in ["metrics", "deps", "outs", "plots"]:
            param_string = str(pipe_params.get(key, {}))
            # find all values within ${} and add them to file_list
            file_list.extend(re.findall(r"\${(.*?)}", param_string))
        file_dict = {}
        for k in file_list:
            if k in flat_params:
                file_dict[k] = flat_params[k]
            elif k == "item":
                file_dict["directory"] = sub_stage
            else:
                raise ValueError(f"File {k} not found in {pipe_params.keys()}")
        file_dict = unflatten_dict(file_dict)
    else:
        pipe_params = dvc.api.params_show(pipeline_file, stages=stage, repo=directory)
        file_dict = unflatten_dict(pipe_params)
    params["files"] = file_dict.pop("files", {})
    params["files"].update(file_dict)
    params["files"]["stage"] = main_stage
    # Merge remaining params
    params = OmegaConf.merge(params, file_dict)
    params = OmegaConf.to_container(OmegaConf.create(params), resolve=True)
    if name is not None:
        params["files"]["name"] = name
    return params


def prepare_files(params_file, stage, params, id_):
    # Turns the dictionary into a FileConfig object.
    # This creates a new directory at files.directory
    # It also creates a new directory at files.directory/files.data_dir
    # It also creates a new directory at files.directory/files.reports_dir
    # If a stage is specified, it also creates a new directory at files.directory/files.reports/stage
    params["files"]["_target_"] = "deckard.base.files.FileConfig"
    tmp_stage = stage.split("@")[0]
    sub_stage = stage.split("@")[1] if tmp_stage != stage else None
    if sub_stage is not None:
        params["files"]["directory"] = sub_stage
    params["files"]["stage"] = tmp_stage
    params["files"]["name"] = (
        id_ if params["files"].get("name", None) is None else params["files"]["name"]
    )
    params["files"]["params_file"] = Path(params_file).name
    # This creates a the object
    files = instantiate(params["files"])
    # Which will return the dictionary of the files
    files = files.get_filenames()
    # If the params_file is in the files, then the params_file is the params_file
    if "params_file" in files:
        params_file = files["params_file"]
    # Otherwise we take the folder of the score_dict_file and change the name to whatever the params_file is
    elif "score_dict_file" in files:
        params_file = Path(files["score_dict_file"]).with_name(params_file)
    else:
        raise ValueError(
            f"Neither params_file nor score_dict_file found in {list(files.keys())}.",
        )

    # Save the params to the params_file
    Path(params_file).parent.mkdir(exist_ok=True, parents=True)
    with Path(params_file).open("w") as f:
        yaml.dump(params, f)
    return files


def get_stages(pipeline_file="dvc.yaml", stages=None, repo=None):
    try:
        with open(pipeline_file, "r") as f:
            def_stages = yaml.safe_load(f)["stages"].keys()
    except NotGitRepository:
        raise ValueError(
            f"Directory {repo} is not a dvc repository. Please run `dvc init` in {repo} and try again.",
        )
    if stages is None or stages == []:
        logger.info("No stages specified. Running default from hydra configuration")
        stages = [None]
    elif isinstance(stages, str):
        stages = [stages]
    else:
        assert isinstance(stages, list), f"args.stage is of type {type(stages)}"
        for stage in stages:
            tmp_stage = stage.split("@")[0]
            assert (
                tmp_stage in def_stages
            ), f"Stage {stage} not found in {pipeline_file}. Available stages: {def_stages}"
    return stages


def get_params_from_disk(
    params_file,
    pipeline_file,
    directory,
    stage,
    config_dir,
    config_file,
):
    if stage is not None:
        params = get_dvc_stage_params(
            stage=stage,
            params_file=params_file,
            pipeline_file=pipeline_file,
            directory=directory,
        )
    else:
        # Use hydras compose to get the params
        assert config_dir is not None, "config_dir must be specified if stage is None"
        with initialize_config_dir(
            config_dir=config_dir,
            job_name=Path(config_file).stem,
            version_base="1.3",
        ):
            cfg = compose(config_name=config_file)
        params = OmegaConf.to_container(cfg, resolve=True)
        params["files"] = dict(params.pop("files", params))
        params["files"]["_target_"] = "deckard.base.files.FileConfig"
        params["files"]["stage"] = None
        params["stage"] = None
    return params


def run_stage(
    params_file="params.yaml",
    pipeline_file="dvc.yaml",
    directory=".",
    stage=None,
    overrides=None,
    config_dir=None,
    config_file=None,
    sub_dict=None,
):
    logger.info(
        f"Running stage {stage} with params_file: {params_file} and pipeline_file: {pipeline_file} in directory {directory}",
    )
    params = get_params_from_disk(
        params_file,
        pipeline_file,
        directory,
        stage,
        config_dir,
        config_file,
    )
    params = add_overrides(overrides, params)
    if sub_dict is None:
        params["_target_"] = "deckard.experiment.Experiment"
        exp = instantiate(params)
        id_ = exp.name
        files = prepare_files(params_file, stage, params, id_)
        score = exp(**files)
    else:
        possible_subdicts = ["data", "model", "attack", "scorers", "plots", "files"]
        assert (
            sub_dict in possible_subdicts
        ), f"sub_dict must be one of {possible_subdicts}"
        target = f"deckard.{sub_dict}.{sub_dict.capitalize()}"
        params["_target_"] = target
        exp = instantiate(params[sub_dict])
        id_ = exp.name
        files = params["files"]
        params[sub_dict]["files"] = files
        files = prepare_files(params_file, stage, params[sub_dict], id_)
        score = exp(**files)
    return id_, score


def add_overrides(overrides, params):
    old_params = deepcopy(params)
    if overrides is not None and len(overrides) > 0:
        # convert from dot notation to nested dict
        overrides = OmegaConf.from_dotlist(overrides)
        params = OmegaConf.merge(params, overrides)
        params = OmegaConf.to_container(params, resolve=True)
        assert (
            params != old_params
        ), f"Params are the same as before overrides: {overrides}"
    params = OmegaConf.create(params)
    params = OmegaConf.to_container(params, resolve=True)
    return params


def run_stages(
    stages,
    pipeline_file="dvc.yaml",
    params_file="params.yaml",
    repo=None,
    config_dir=None,
    config_file=None,
    sub_dict=None,
):
    results = {}
    stages = get_stages(stages=stages, pipeline_file=pipeline_file, repo=repo)
    for stage in stages:
        id_, score = run_stage(
            stage=stage,
            pipeline_file=pipeline_file,
            params_file=params_file,
            directory=repo,
            config_dir=config_dir,
            config_file=config_file,
            sub_dict=sub_dict,
        )
        results[id_] = score
    return results
