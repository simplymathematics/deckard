import argparse
import logging
import os
import yaml
import json
import subprocess
import hydra
import dvc.api
from pathlib import Path
from omegaconf import DictConfig
from art.utils import to_categorical
from deckard.base import Experiment
from deckard.layers.parse import parse

logger = logging.getLogger(__name__)





def run_dvc_experiment(stage = None, params = None):
    results = {}
    # Load params from dvc
    if stage is None:
        with open(Path(os.getcwd(), "dvc.yaml"), "r") as f:
            stages = yaml.load(f, Loader=yaml.FullLoader)["stages"].keys()
        stage = list(stages)[-1]
    params = dvc.api.params_show("params.yaml", stages=[stage]) if params is None else params
    # Update params with paths from dvc
    params["files"]["data_file"] = str(Path(stage, params["files"]["data_file"]))
    params["files"]["model_file"] = str(Path(stage, params["files"]["model_file"]))
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
    results[stage] = exp.run()
    return results

def run_queue(inputs, stage):
    results = {}
    failures = 0
    successes = 0
    total = len(inputs)
    while len(inputs) > 0:
        pipeline = inputs.pop(0)
        if pipeline.name != "params.yaml":
            # move file to "params.yaml"
            Path("params.yaml").unlink(missing_ok=True)
            Path(pipeline).rename("params.yaml")
        try:
            output = run_dvc_experiment(stage)
            successes += 1
            results[str(pipeline)] = output
        except Exception as e:
            output = str(e)
            failures += 1
            raise e
        finally:
            logger.info(f"Successes: {successes}, Failures: {failures}, Total: {total}")
    return results
    
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_path", type=str, default="queue")
    parser.add_argument("--regex", type=str, default="*.yaml")
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("stage", type=str, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    files = {}
    inputs = list(Path(os.getcwd(), args.queue_path).glob(args.regex))
    failures = 0
    successes = 0
    total = len(inputs)
    if total == 0:
        if Path("params.yaml").exists():
            inputs = [Path("params.yaml")]
            total = 1
        else:
            raise ValueError(
                f"No experiments found in queue folder {args.queue_path} with regex {args.regex} and no params.yaml file in current directory.",
            )
    elif Path("params.yaml").exists():
        Path("params.yaml").unlink()
    results = run_queue(inputs, args.stage)
    logger.info(json.dumps(results))