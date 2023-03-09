import logging
import os
import yaml
import json
from pathlib import Path
from typing import Union, List
import hydra
from omegaconf import DictConfig, OmegaConf
from deckard.base import Experiment
from deckard.layers.runner import load_dvc_experiment
from deckard.layers.parse import generate_paths_from_params
from deckard.base.hashable import my_hash

logger = logging.getLogger(__name__)
def merge_params(default, params) -> dict:
    """
    Overwrite default params with params if key is found in default.
    :param default: Default params
    :param params: Params to overwrite default params
    :return: Merged params
    """
    for key, value in params.items():
        if key in default:
            if isinstance(default[key], dict):
                default[key] = merge_params(default[key], value)
            else:
                default[key] = value
        else:
            logger.warning(f"Key {key} not found in default params. Ignoring.")
    return default

def parse_stage(stage:Union[dict, List[dict]] = None,  params:dict = None) -> dict:
    """
    Parse params from dvc.yaml and merge with params from hydra config
    :param stage: Stage to load params from. If None, loads the last stage in dvc.yaml
    :param params: Params to merge with params from dvc.yaml
    :return: Merged params
    """
    # Load params from dvc
    if stage is None:
        with open(Path(os.getcwd(), "dvc.yaml"), "r") as f:
            stages = yaml.load(f, Loader=yaml.FullLoader)["stages"].keys() #  Get all stages
        stage = list(stages)[-1] # Get the last stage
    if params is None:
        with open(Path(os.getcwd(), "paramns.yaml"), "r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open(Path(os.getcwd(), "dvc.yaml"), "r") as f:
            key_list = yaml.load(f, Loader=yaml.FullLoader)["stages"][stage]["params"]
        with open(Path(os.getcwd(), "params.yaml"), "r") as f:
            all_params = yaml.load(f, Loader=yaml.FullLoader)
        default_params = read_subset_of_params(key_list, all_params)
        params = merge_params(default_params, params)
    # Update params with paths from dvc
    full_report = params["files"]["path"]
    parents = list(Path(full_report).parents)
    name = Path(full_report).name
    parents.insert(1, Path(stage))
    params["files"]["path"] = str(Path(params["files"]["reports"], *parents, name))
    
    return params

def load_dvc_experiment(stage = None, params = None) -> Experiment:
    """
    Load experiment from dvc.yaml for a given stage and overwrite the default params with optional user-supplied params.
    :param stage: Stage to load params from. If None, loads the last stage in dvc.yaml
    :param params: Params to merge with params from dvc.yaml
    :return: Experiment
    """
    # Load and run experiment from yaml
    params = parse_stage(stage, params)
    tag = "!Experiment:"
    yaml.add_constructor(tag, Experiment)
    exp = yaml.load(f"{tag}\n" + str(params), Loader=yaml.FullLoader)
    return exp



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
    if "dry_run" in cfg:
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
    logging.basicConfig(level=verbosity)
    if not Path(os.getcwd(), queue).exists():
        Path(os.getcwd(), queue).mkdir()
    params = OmegaConf.to_container(cfg, resolve=True)
    params = generate_paths_from_params(params, default=None) # This is a hack to add file names based on the hash of the parameterization    
    logger.debug("Params:\n"+json.dumps(params, indent=4)) # For debugging
    filename = Path(os.getcwd(), queue, my_hash(params)+".yaml") # This is the file that will be used to run the experiment
    with open(filename, 'w') as f:
        yaml.dump(params, f)
    if not dry_run: #If dry_run is true, this will just write the parameters to a file and not run the experiment
        exp =  load_dvc_experiment(stage=stage, params=params)
        ########################################
        # For # This will run the experiment and return a dictionary of results. 
        # This uses the normal fit/predict/eval loop and returns the scores 
        # on the test set as specified in the config file. 
        # So, essentially, we would a function that takes in the parameters and returns the score here.
        try:
            results = exp.run()
            logger.info("Results:\n"+json.dumps(results, indent=4))
            with open(results['scores'], 'r') as f:
                score = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Score:\n"+json.dumps(score, indent=4))
            score = list(score.values())[0]
            logger.info("Score:\n"+json.dumps(score, indent=4))
            with open(results['time'], 'r') as f:
                time = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Time:\n"+json.dumps(time, indent=4))
            time = list(time.values())[0]
            logger.info("Time:\n"+json.dumps(time, indent=4))
        except Exception as e:
            raise e
            # score = None
        ########################################
    else:
        score = 0
    return score


def read_subset_of_params(key_list:list, params:dict):
    new_params = {}
    for key in key_list:
        if key in params:
            new_params[key] = params[key]
        elif "." in key:
            first_loop = True
            dot_key = key
            total = len(key.split("."))
            i = 1
            for entry in key.split("."):
                if first_loop is True:
                    sub_params = params
                    first_loop = False
                if entry in sub_params:
                    sub_params = sub_params[entry]
                    i += 1
                else:
                    raise ValueError(f"{dot_key} not found in {params}")
                if i == total:
                    new_params[dot_key.split(".")[0]] = {**sub_params}
                else:
                    pass
                
                    
        else:
            raise ValueError(f"{key} not found in {params}")
    return new_params
            
    
    
if '__main__' == __name__:
    logging.basicConfig(level="INFO")
    hydra_optimizer()