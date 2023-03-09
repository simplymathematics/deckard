import argparse
import logging
import os
import subprocess
import yaml
import json
from pathlib import Path
import hydra
from optuna.trial import Trial
import optuna
from omegaconf import DictConfig, OmegaConf
from deckard.base import Experiment
from deckard.layers.runner import load_dvc_experiment
from deckard.layers.parse import parse
from deckard.base.hashable import my_hash
logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path=str(Path(os.getcwd(), "conf")),
    config_name="config",
)
def hydra_optimizer(cfg:DictConfig):
    # Stage selects a subset of the pipeline to run (i.e. number of layers to run inside a single container)
    if "stage" in cfg:
        stage = cfg.stage
        del cfg.stage
    else:
        stage = None
    if "filename" in cfg:
        dry_run = cfg.dry_run
        del cfg.dry_run
    else:
        dry_run = False
    if "queue_path" in cfg:
        queue = cfg.queue_path
        del cfg.queue_path
    else:
        queue = "queue"
    if "verbosity" in cfg:
        verbosity = cfg.verbosity
        del cfg.verbosity
    else:
        verbosity = "INFO"
    if "filename" in cfg:
        filename = cfg.filename
        del cfg.filename
    else:
        filename = "tmp.yaml"
    logging.basicConfig(level=verbosity)
    if not Path(os.getcwd(), queue).exists():
        Path(os.getcwd(), queue).mkdir()
    params = OmegaConf.to_container(cfg, resolve=True)
    params = parse(params,) # This is a hack to add file names based on the hash of the parameterization    
    with open(Path(os.getcwd(), "dvc.yaml"), "r") as f:
        stages = yaml.load(f, Loader=yaml.FullLoader)
    new_pipe = {}
    if stage is None:
        new_pipe['stages'] = stages['stages']
    else:
        new_pipe['stages'][stage] = stages['stages'][stage]
    big_dict = {}
    Path("pipelines").mkdir(exist_ok=True)
    for k,v in params.items():
        if k == "files":
            new_dict = {}
            for file in params[k]:
                if file in ["data_file", "model_file", "attack_file", "reports", "path"]:
                    new_dict[file] = str(Path("..", params[k][file]).as_posix())
            v = new_dict
        big_dict[k] = v
    # import pandas as pd
    # big_dict = pd.json_normalize(big_dict)
    # big_dict = big_dict.to_dict(orient='records')[0]
    # new_list = []
    # for entry in big_dict.items():
    #     new_list.append({entry[0] : entry[1]})
    # print(big_dict)
    
    input("Press Enter to continue...")
    # new_pipe['vars'] = new_list
    tmp_file = Path("pipelines", "dvc.yaml")
    with open(tmp_file, "w") as f:
        yaml.dump(new_pipe, f)
    with open(Path("pipelines", "params.yaml"), "w") as f:
        yaml.dump(big_dict, f)
    
    
if '__main__' == __name__:
    
    hydra_optimizer()