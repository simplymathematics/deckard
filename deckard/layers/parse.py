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
    config_path=str(Path(os.getcwd(), "conf")),
    config_name="config",
)
def hydra_parser(cfg: DictConfig, **kwargs):
    params = OmegaConf.to_object(cfg)
    params = generate_paths_from_params(params, **kwargs)
    print(yaml.dump(params, indent=4))
    logger.info("Successfully parsed the hydra config and saved it to params.yaml")

def generate_paths_from_params(params:dict, queue_path=None, default = "params.yaml", filename = "default.yaml") -> dict:
    """
    Parse the hydra config and save it to a yaml file in both the queue_path and the current working directory with the name "params.yaml".
    :param params: The hydra config as a dictionary
    :param queue_path: The path to the queue directory
    :param default: The name of the default yaml file
    :param filename: The name of the yaml file to save the parsed config to
    :return: The parsed config as a dictionar
    """
    # For convenience, we'll make a copy of the params
    params = dict(params)
    files = dict(params["files"])  
    # Setup the data 
    data = dict(params["data"])
    data['path'] = params['data']['path'] if params['data']['path'] is not None else my_hash(data)
    data['filetype'] = params['data']['filetype'] if params['data']['filetype'] is not None else "npz"
    data['filename'] = params['data']['filename'] if params['data']['filename'] is not None else my_hash(data)
    files['data_file'] = str(Path(data['path'], data['filename'] + "." + data['filetype']).as_posix())
    del data['filename']
    del data['path']
    del data['filetype']
    params['data'] = data
    # Setup the model
    model = dict(params["model"])
    model['path'] = params['model']['path'] if params['model']['path'] is not None else my_hash(model)
    model['filetype'] = params['model']['filetype'] if params['model']['filetype'] is not None else "pkl"
    model['filename'] = params['model']['filename'] if params['model']['filename'] is not None else my_hash(model)
    files['model_file'] = str(Path(model['path'], model['filename'] + "." + model['filetype']).as_posix())
    del model['path']
    del model['filename']
    del model['filetype']
    params['model'] = model
    # Setup the attack
    attack = dict(params["attack"])
    attack['path'] = params['attack']['path'] if params['attack']['path'] is not None else my_hash(attack)
    attack['filetype'] = params['attack']['filetype'] if params['attack']['filetype'] is not None else "pkl"
    attack['filename'] = params['attack']['filename'] if params['attack']['filename'] is not None else my_hash(attack) 
    files['attack_file'] = str(Path(attack['path'], attack['filename'] + "." + attack['filetype']).as_posix())
    del attack['path']
    del attack['filename']
    del attack['filetype']
    params['attack'] = attack
    # Setup the output
    params['files'] = files
    params["files"]["path"] = str(my_hash(params)) if params["files"]["path"] is None else params["files"]["path"]
    params = DictConfig(params)
    params = OmegaConf.to_container(params, resolve=True)
    return params

def dump_params_to_queue(params:dict, queue_path=None, filename = None) -> None:
    # Write the params to the queue
    if queue_path is not None:
        filename = Path(queue_path, my_hash(params) + ".yaml") if filename is None else Path(queue_path, filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(Path(filename), "w") as f:
            yaml.dump(params, f)
    logger.info(f"Wrote params to {filename}")
    return None     
            
def dump_params_to_default(params:dict, default:str = "params.yaml") -> None:
    # Write the params to the default
    if default is not None:
        if Path(default).exists():
            Path(os.getcwd(), default).unlink()
        with open(Path(default), "w") as f:
            yaml.dump(params, f)
    logger.info(f"Wrote params to {default}")
    return None



def generate_stage_from_params(stage, params:dict, filename="dvc.yaml", output_folder="reports",) -> None:
    """
    Generate a stage from the params
    :param stage: The stage to generate
    :param params: The params to generate the stage from
    :param filename: The name of the file to save the stage to
    :return: None
    """
    # Load the dvc.yaml file
    with open(Path(filename), "w") as f:
        pipe = yaml.load(f, Loader=yaml.FullLoader)['stages'][stage]
    # If there is a dictionary inside of the params, we need to flatten it
    flattened = flatten_dict(params)
    # Replace the params in the dvc.yaml (denoted by "${ }") file with the flattened params
    for key, value in flattened.items():
        for ki, vi in pipe.items():
            if isinstance(vi, str):
                pipe[ki] = vi.replace("${" + key + "}", str(value))
        
    
def flatten_dict():
    pass

def generate_grid():
    pass
if "__main__" == __name__:
    _ = hydra_parser()
    assert Path("params.yaml").exists(), \
        f"Params path, 'params.yaml', does not exist. Something went wrong."
    sys.exit(0)
