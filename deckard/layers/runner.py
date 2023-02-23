import argparse
import logging
import os
import yaml
import dvc.api
from pathlib import Path
from deckard.base import Experiment

logger = logging.getLogger(__name__)


def load_dvc_experiment(stage = None, filename = "params.yaml", params = None):
    # Load params from dvc
    if stage is None:
        with open(Path(os.getcwd(), "dvc.yaml"), "r") as f:
            stages = yaml.load(f, Loader=yaml.FullLoader)["stages"].keys()
        stage = list(stages)[-1]
    if params is None:
        params = dvc.api.params_show(Path(filename).as_posix(), stages=[stage])
    # Update params with paths from dvc
    if "@" in stage:
        stage = stage.split("@")[0]
    params["files"]["data_file"] = str(Path(stage, params["files"]["data_file"])) #if "data_file" not in params["files"] else params["files"]["data_file"]
    params["files"]["model_file"] = str(Path(stage, params["files"]["model_file"])) #if "model_file" not in params["files"] else params["files"]["model_file"]
    Path(params["files"]["data_file"]).parent.mkdir(parents=True, exist_ok=True)
    Path(params["files"]["model_file"]).parent.mkdir(parents=True, exist_ok=True)
    full_report = params["files"]["path"]
    parents = list(Path(full_report).parents)
    name = Path(full_report).name
    parents.insert(1, Path(stage))
    params["files"]["path"] = str(Path(params["files"]["reports"], *parents, name))
    # Load and run experiment from yaml
    tag = "!Experiment:"
    yaml.add_constructor(tag, Experiment)
    exp = yaml.load(f"{tag}\n" + str(params), Loader=yaml.FullLoader)
    
    return exp

    
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("stage", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    exp =  load_dvc_experiment(args.stage, filename = "params.yaml")
    results = exp.run()
    print(yaml.dump(results))
    