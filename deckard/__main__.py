""""Runs a submodule passed as an arg."""

import argparse
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
layer_list = list(Path(Path(__file__).parent, "layers").glob("*.py"))
layer_list = [layer.stem for layer in layer_list]
if "__init__" in layer_list:
    layer_list.remove("__init__")


def run_submodule(submodule, args):
    cmd = f"python -m deckard.layers.{submodule} {args}"
    logger.info(f"Running {cmd}")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as proc:
        for line in proc.stdout:
            print(line.rstrip().decode("utf-8"))
        if len(proc.stderr.readlines()) > 0:
            logger.error(f"Error running {cmd}")
            for line in proc.stderr:
                logger.error(line.rstrip().decode("utf-8"))
            return 1
        else:
            return 0


def parse_and_repro(args):
    cmd = f"python -m deckard.layers.parse {args}"
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as proc:
        for line in proc.stdout:
            print(line.rstrip().decode("utf-8"))
        if len(proc.stderr.readlines()) > 0:
            raise ValueError(f"Error parsing with options {args}")
    if Path(Path(), "dvc.yaml").exists():
        cmd = f"dvc repro"
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as proc:
            for line in proc.stdout:
                print(line.rstrip().decode("utf-8"))
            if len(proc.stderr.readlines()) > 0:
                logger.error(f"Error running {cmd}")
                for line in proc.stderr:
                    logger.error(line.rstrip().decode("utf-8"))
                raise ValueError(f"Error running dvc.yaml")
    else:
        raise ValueError("No dvc.yaml file found. Please construct a pipeline.")
    return 0


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "submodule",
        type=str,
        nargs="?",
        help=f"Submodule to run. Choices: {layer_list}",
    )
    parser.add_argument("other_args", type=str, nargs="*")
    args = parser.parse_args()
    submodule = args.submodule
    if submodule not in layer_list and submodule is not None:
        raise ValueError(f"Submodule {submodule} not found. Choices: {layer_list}")
    other_args = " ".join(args.other_args)
    if submodule is None:
        assert parse_and_repro(other_args) == 0
    else:
        assert run_submodule(submodule, other_args) == 0
