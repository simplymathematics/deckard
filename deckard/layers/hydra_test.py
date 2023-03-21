import logging
import os
import sys
import hydra
import yaml
import flatdict
from pathlib import Path
import pandas as pd
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
def hydra_parser(cfg: DictConfig):
    params = OmegaConf.to_object(cfg)
    params = generate_paths_from_params(params)
    grid = generate_grid(params)
    i = 0
    if "queue_path" in cfg:
        queue_path = cfg["queue_path"]
        del cfg.queue_path
    else:
        queue_path = "queue"
    if "default" in cfg:
        default = cfg["default"]
        del cfg.default
    else:
        default = "params.yaml"
    for entry in grid:
        print(f"Entry {i+1} of {len(grid)}")
        i += 1
        param_dict = flatten_dict(entry)
        dump_params_to_default(params, default)
        stage_dict = generate_stage_from_params("train", param_dict)
        dump_stage_to_queue(stage_dict, queue_path=queue_path)
        dump_params_to_queue(params, queue_path)
        print(yaml.dump(params, indent=4))
        logger.info("Successfully parsed the hydra config and saved it to params.yaml")

def generate_paths_from_params(params:dict) -> dict:
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

def dump_params_to_queue(params:dict, queue_path=None, filename = None, return_path = False) -> None:
    """
    Takes the params and dumps them to the queue_path. Generates a unique filename based on the hash of the params if filename is None.
    :param params: The params to dump to the queue
    :param queue_path: The path to the queue
    :param filename: The filename to save the params to
    :param return_path: Whether to return the path to the params
    :return: None
    """
    # Write the params to the queue
    if queue_path is not None:
        filename = Path(queue_path, my_hash(params) + ".yaml") if filename is None else Path(queue_path, filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(Path(filename), "w") as f:
            yaml.dump(params, f)
        assert Path(filename).exists(), f"Failed to write params to {filename}"
    logger.info(f"Wrote params to {filename}")
    if return_path is True:
        return filename
    else:
        return None     
            
def dump_params_to_default(params:dict, default:str = "params.yaml", return_path = False) -> None:
    """
    Takes the params and dumps them to the default path. Generates a unique filename based on the hash of the params if filename is None.
    :param params: The params to dump to the queue
    :param default: The path to the default params
    :param return_path: Whether to return the path to the params
    :return: None
    """
    # Write the params to the default
    if default is not None:
        if Path(default).exists():
            Path(os.getcwd(), default).unlink()
        with open(Path(default), "w") as f:
            yaml.dump(params, f)
    logger.info(f"Wrote params to {default}")
    if return_path is True:
        return default
    else:
        return None

def dump_stage_to_queue(stage:dict, queue_path=None, filename = None, return_path = False) -> None:
    """
    Takes the stage and dumps it to the queue_path. Generates a unique filename based on the hash of the stage if filename is None.
    :param stage: The stage to dump to the queue
    :param queue_path: The path to the queue
    :param filename: The filename to save the stage to
    :param return_path: Whether to return the path to the stage
    :return: None
    """
    # Write the stage to the queue
    if queue_path is not None:
        filename = Path(queue_path, my_hash(stage) + ".yaml") if filename is None else Path(queue_path, filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(Path(filename), "w") as f:
            yaml.dump(stage, f)
        assert Path(filename).exists(), f"Failed to write stage to {filename}"
    logger.info(f"Wrote stage to {filename}")
    if return_path is True:
        return filename
    else:
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
    with open(Path(filename), "r") as f:
        pipe = yaml.load(f, Loader=yaml.FullLoader)['stages'][stage]
    # If there is a dictionary inside of the params, we need to flatten it
    flattened = flatten_dict(params)
    # Replace the params in the dvc.yaml (denoted by "${ }") file with the flattened params
    pipe = replace_template(pipe, flattened)
    # Write the stage to the dvc.yaml file
    new_filename = Path(stage, my_hash(params) + ".yaml")
    new_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(new_filename, "w") as f:
        yaml.dump(pipe, f)
    return pipe
    
def replace_template(pipe, params) -> dict:
    """
    Replace the template in the pipe with the params.
    :param pipe: The pipe to replace the template in
    :param params: The params to replace the template with
    :return: The pipe with the template replaced
    """
    for key, value in params.items():
        for ki, vi in pipe.items():
            if isinstance(vi, str):
                pipe[ki] = vi.replace(r"${" + key + r"}", str(value))
            elif isinstance(vi, list):
                i = 0
                for v in vi:
                    if isinstance(v, str):
                        vi[i] = v.replace(r"${" + key + r"}", str(value))
                    i += 1
                pipe[ki] = vi
            elif isinstance(vi, dict):
                for k,v in vi.items():
                    if isinstance(v, str):
                        vi[k] = v.replace(r"${" + key + r"}", str(value))
                    else:
                        vi[k] = replace_template(v, params)
                pipe[ki] = vi
    return pipe
    
def flatten_dict(param_dict, sep='.') -> dict:
    """
    Flatten a dictionary with nested dictionaries, using a separator to denote the nesting.
    :param param_dict: The dictionary to flatten
    :param sep: The separator to use to denote the nesting
    :return: The flattened dictionary
    """
    new_dict = {}
    [new_dict] = pd.json_normalize(param_dict, sep=sep).to_dict(orient='records')
    return new_dict

def generate_grid(param_dict) -> list:
    """
    Return a grid of all possible combinations of the parameters.
    :param param_dict: The dictionary of parameters to generate the grid from
    :return: A list of dictionaries, each dictionary is a possible combination of the parameters
    """
    new_dict = {}
    for key, value in param_dict.items():
        if not isinstance(value, list):
            new_dict[key] = [value]
        else:
            new_dict[key] = value
    grid = list(ParameterGrid(new_dict))
    return grid


if "__main__" == __name__:
    _ = hydra_parser()
    assert Path("params.yaml").exists(), \
        f"Params path, 'params.yaml', does not exist. Something went wrong."
    sys.exit(0)
