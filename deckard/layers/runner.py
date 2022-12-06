import argparse
import logging
import os
from pathlib import Path

import dvc.api
import hydra
import yaml
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from deckard.base import Experiment
from deckard.base.hashable import my_hash


logger = logging.getLogger(__name__)

def run_experiment(params, stage):
    tag = "!Experiment:"
    yaml.add_constructor(tag, Experiment)
    params["files"]['data_file'] = str(Path(stage, params["files"]['data_file']))
    params['files']['model_file'] = str(Path(stage, params["files"]['model_file']))
    Path(params['files']['data_file']).parent.mkdir(parents=True, exist_ok=True)
    Path(params['files']['model_file']).parent.mkdir(parents=True, exist_ok=True)
    full_report = params['files']['path']
    print(full_report)
    parents = list(Path(full_report).parents)
    name = Path(full_report).name
    parents.insert(1, Path(stage))
    params['files']['path'] = str(Path(params['files']['reports'], *parents, name))
    print(params['files']['data_file'])
    print(params['files']['model_file'])
    print(params['files']['path'])
    exp = yaml.load(f"{tag}\n" + str(params), Loader=yaml.FullLoader)
    files = exp.run()
    return files

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_path", type=str, default="queue")
    parser.add_argument("--regex", type=str, default="*.yaml")
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("stage", type=str, default="train")
    args = parser.parse_args()
    logging.basicConfig(level=args.verbosity)
    with open(Path(os.getcwd(), "dvc.yaml"), "r") as f:
        stages = yaml.load(f, Loader=yaml.FullLoader)['stages'].keys()
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
            raise ValueError(f"No experiments found in queue folder {args.queue_path} with regex {args.regex} and no params.yaml file in current directory.")
    elif Path("params.yaml").exists():
        Path("params.yaml").unlink()
    while len(inputs) > 0:
        pipeline = inputs.pop(0)
        try:
            Path(pipeline).rename(Path("params.yaml"))
        except FileNotFoundError:
            pass
        logger.info(f"Running experiment {pipeline} with {len(inputs)} experiments remaining.")

        files[str(pipeline)] = {}
        params = dvc.api.params_show("params.yaml", stages = [args.stage])
        try:
            files[str(pipeline)]= run_experiment(params, stage = args.stage)
            successes += 1
        except Exception as e:
            logger.warning("Experiment failed. Moving to failed folder.")
            Path("params.yaml").rename(Path("failed", str(pipeline)))
            failures += 1
            raise e
        finally:
            assert failures + successes == total, f"Failures {failures} + successes {successes} != total {total}"
            logger.info(f"Finished {len(files)} of {total} experiments with {failures} failures.")



