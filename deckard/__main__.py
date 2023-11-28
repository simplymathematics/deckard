""""Runs a submodule passed as an arg."""

import argparse
import subprocess
import logging
from pathlib import Path
from omegaconf import OmegaConf
from .layers.parse import save_params_file

OmegaConf.register_new_resolver("eval", eval)

logger = logging.getLogger(__name__)
layer_list = list(Path(Path(__file__).parent, "layers").glob("*.py"))
layer_list = [layer.stem for layer in layer_list]
if "__init__" in layer_list:
    layer_list.remove("__init__")
layer_list.append(None)


def run_submodule(submodule, args):
    if len(args) == 0:
        cmd = f"python -m deckard.layers.{submodule}"
    else:
        cmd = f"python -m deckard.layers.{submodule} {args}"
    logger.info(f"Running {cmd}")
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
    ) as proc:
        for line in proc.stdout:
            print(line.rstrip().decode("utf-8"))
        if proc.returncode != 0:
            logger.error(f"Error running {cmd}")
            for line in proc.stderr:
                logger.error(line.rstrip().decode("utf-8"))
            return 1
        else:
            return 0


def parse_and_repro(args, default_config="default.yaml", config_dir="conf"):
    if len(args) == 0:
        assert (
            save_params_file(
                config_dir=Path(Path(), config_dir)
                if not Path(config_dir).is_absolute()
                else Path(config_dir),
                config_file=default_config,
            )
            is None
        )
        assert Path(Path(), "params.yaml").exists()
    else:
        cmd = f"python -m deckard.layers.parse {args} --config_file {default_config}"
        # error = f"error parsing command: {cmd} {args}"
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True,) as proc:
            for line in proc.stdout:
                print(line.rstrip().decode("utf-8"))
    if Path(Path(), "dvc.yaml").exists():
        cmd = "dvc repro"
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True) as proc:
            for line in proc.stdout:
                print(line.rstrip().decode("utf-8"))

    else:
        raise ValueError("No dvc.yaml file found. Please construct a pipeline.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submodule", type=str, help=f"Submodule to run. Choices: {layer_list}",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="default hydra configuration file that you would like to reproduce with dvc repro.",
    )
    parser.add_argument("--config_dir", type=str, default="conf")
    parser.add_argument("other_args", type=str, nargs="*")
    args = parser.parse_args()
    submodule = args.submodule
    if submodule is not None:
        assert (
            args.config_file is None
        ), "config_file and submodule cannot be specified at the same time"
    if submodule not in layer_list and submodule is not None:
        raise ValueError(f"Submodule {submodule} not found. Choices: {layer_list}")
    if len(args.other_args) > 0:
        other_args = " ".join(args.other_args)
    else:
        other_args = []
    if submodule is None:
        assert (
            parse_and_repro(other_args, args.config_file, config_dir=args.config_dir)
            == 0
        )
    else:
        assert run_submodule(submodule, other_args) == 0
