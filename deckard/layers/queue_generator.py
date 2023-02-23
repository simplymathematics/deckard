import argparse
import logging
import os
import shutil
import yaml
import dvc.api
from pathlib import Path
from random import shuffle
from deckard.base import Experiment
from deckard.layers.runner import load_dvc_experiment

logger = logging.getLogger(__name__)

def run_queue(inputs, stage):
    results = {}
    failures = 0
    successes = 0
    total = len(inputs)
    while len(inputs) > 0:
        pipeline = inputs.pop(0)
        # if pipeline.name != "params.yaml":
        #     # move file to "params.yaml"
        #     Path("params.yaml").unlink(missing_ok=True)
        #     shutil.copy(pipeline, Path("params.yaml"))
        try:
            exp = load_dvc_experiment(stage, filename=pipeline)
            results = exp.run()
            successes += 1
            results[str(pipeline)] = results
        except Exception as e:
            raise e
        finally:
            logger.info(f"Successes: {successes}, Failures: {failures}, Total: {total}")
    return results
    
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue_path", type=str, default=None)
    parser.add_argument("--regex", type=str, default="*.yaml")
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("stage", type=str, default=None)
    args = parser.parse_args()
    queue_path = args.queue_path if args.queue_path is not None else args.stage
    logging.basicConfig(level=args.verbosity)
    files = {}
    inputs = list(Path(os.getcwd(), queue_path).glob(args.regex))
    shuffle(inputs)
    failures = 0
    successes = 0
    total = len(inputs)
    if total == 0:
        if Path("params.yaml").exists():
            inputs = [Path("params.yaml")]
            total = 1
        else:
            raise ValueError(
                f"No experiments found in queue folder {queue_path} with regex {args.regex} and no params.yaml file in current directory.",
            )
    results = run_queue(inputs, args.stage)