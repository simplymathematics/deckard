import logging
import os
import sys
import dvc.api
import hydra
import yaml
import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import ParameterGrid
from deckard.base.hashable import my_hash

config_path = Path(os.getcwd())

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=Path(os.getcwd(), "conf"),
    config_name="config",
)
def hydra_parser(cfg: DictConfig, **kwargs):
    params = OmegaConf.to_object(cfg)
    params = parse(params, **kwargs)
    print(json.dumps(params, indent=4))
    logger.info("Successfully parsed the hydra config and saved it to params.yaml")

def parse(params:dict, queue_path="queue", default = "params.yaml", filename = "default.yaml"):
    params = dict(params)
    files = dict(params["files"])
    data = dict(params["data"])
    model = dict(params["model"])
    if "data" in params and "files" in params["data"]:
        files.update(params["data"].pop("files"))
    if "model" in params and "files" in params["model"]:
        files.update(params["model"].pop("files"))
    if "data_file" not in params and "data" in params:
        params["files"]["data_file"] = str(
            Path(
                files["data_path"],
                my_hash(data) + "." + files["data_filetype"],
            ).as_posix(),
        )
    if "model_file" not in params and "model" in params:
        params["files"]["model_file"] = str(
            Path(
                files["model_path"],
                my_hash(model) + "." + files["model_filetype"],
            ).as_posix(),
        )
    if "files" in params:
        params["files"]["path"] = str(my_hash(params))
    if "attack" in params:
        if (
            "files" in params["attack"]
            and "attack_samples_file" in params["attack"]["files"]
        ):
            attack_files = params["attack"].pop("files")
            for atk_file in attack_files:
                params["files"][atk_file] = str(Path(attack_files[atk_file]))
    Path(os.getcwd(), default).unlink()
    filename = Path(queue_path, my_hash(params) + ".yaml") if filename is None else Path(queue_path, filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(filename), "w") as f:
        yaml.dump(params, f)
    with open(Path(default), "w") as f:
        yaml.dump(params, f)
    logger.info(f"Wrote params to {filename} and {default}")
    return params


if "__main__" == __name__:
    _ = hydra_parser()
    assert Path("params.yaml").exists(), \
        f"Params path, 'params.yaml', does not exist. Something went wrong."
    sys.exit(0)
