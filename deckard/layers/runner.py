import argparse
import logging
from pathlib import Path

import dvc.api
import yaml

from deckard.base import Experiment

# from deckard.base.hashable import generate_queue, my_hash, sort_queue

layers = {
    "fit": {
        "fit": True,
        "predict": True,
        "score": True,
        "visualise": True,
        "attack": False,
    },
    "data": {
        "fit": False,
        "predict": False,
        "score": False,
        "visualise": False,
        "attack": False,
    },
    "predict": {
        "fit": False,
        "predict": False,
        "score": False,
        "visualise": False,
        "attack": False,
    },
    "evaluate": {
        "fit": True,
        "predict": True,
        "score": True,
        "visualise": False,
        "attack": False,
    },
    "visualise": {
        "fit": False,
        "predict": False,
        "score": False,
        "visualise": True,
        "attack": False,
    },
    "attack": {
        "fit": True,
        "predict": True,
        "score": True,
        "visualise": True,
        "attack": True,
    },
    "all": {
        "fit": True,
        "predict": True,
        "score": True,
        "visualise": True,
        "art": True,
        "attack": True,
    },
}

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, default="train", nargs="?")
    parser.add_argument("--layer", type=str, default="all")
    parser.add_argument("--regex", type=str, default="**/params.yaml", nargs="?")
    parser.add_argument("queue", default=".", type=str, nargs="?")
    args = parser.parse_args()
    params = dvc.api.params_show(stages=[args.stage])
    if "files" in params:
        params["files"]["path"] = str(Path(params["files"]["path"], args.stage))
    if "data" in params:
        params["data"]["files"]["data_path"] = str(
            Path(args.stage, params["data"]["files"]["data_path"]),
        )
    if "model" in params:
        params["model"]["files"]["model_path"] = str(
            Path(args.stage, params["model"]["files"]["model_path"]),
        )
    if "attack" in params:
        params["files"]["attack_path"] = str(
            Path(args.stage, params["files"]["attack_path"]),
        )
    tag = "!Experiment:"
    yaml.add_constructor(tag, Experiment)
    exp = yaml.load(f"{tag}\n" + yaml.dump(params), Loader=yaml.FullLoader)
    run_params = layers[args.layer]
    files = exp.run(**run_params)
    logger.info(f"Files: {files}")
