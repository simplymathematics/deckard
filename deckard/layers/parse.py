import logging
import os
from pathlib import Path

import hydra
import pandas as pd
import yaml
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
def hydra_parser(cfg: DictConfig, stage=None, verbosity = 'INFO'):
    params = OmegaConf.to_object(cfg)
    if "verbosity" in params:
        del params['verbosity']
        logging.basicConfig(level=verbosity)
    stage = params.pop("stage", None)
    default = params.pop("default", "params.yaml")
    parse(params, stage=stage, default=default)
    
def parse(params: dict, stage:str=None, default:str=None):
    """
    """
    params = generate_paths_from_params(params, stage=stage)
    grid = generate_grid(params)
    i = 0
    with open("dvc.yaml", "r") as f:
        pipe = yaml.load(f, Loader=yaml.FullLoader)['stages']
    if stage is None:
        stage = list(pipe.keys())[-1]
    else:
        assert stage in pipe.keys(), f"{stage} not in {list(pipe.keys())}"
    big_list = []
    if default is not None:
        dump_params_to_default(params, default)
        logger.info("Successfully parsed the hydra config and saved it to params.yaml")
    else:
        assert Path("params.yaml").exists(), f"Path params.yaml does not exist"
    for entry in grid:
        logger.info(f"Entry {i+1} of {len(grid)}")
        i += 1
        param_dict = flatten_dict(entry)
        stage_dict = generate_stage_from_params(stage, param_dict)
        # pipefile = dump_stage_to_queue(entry, stage_dict, stagename = stage)
        paramfile = dump_params_to_queue(entry, Path(stage))
        # assert Path(pipefile).exists(), f"Path {pipefile} does not exist"
        # assert Path(paramfile).exists(), f"Path {paramfile} does not exist"
        
        # assert Path(paramfile).parent == Path(pipefile).parent, f"Path {paramfile} and {pipefile} are not in the same directory"
        logger.info(f"Successfully parsed the hydra config and saved it to {Path(params['files']['reports'], params['files']['path'], 'params.yaml')}")
        logger.debug(yaml.dump(params, indent=4))
        
        big_list.append(params)
    df = pd.DataFrame(big_list)
    return df

def generate_paths_from_params(params:dict, stage=str) -> dict:
    """
    Parse the hydra config and save it to a yaml file in both the path and the current working directory with the name "params.yaml".
    :param params: The hydra config as a dictionary
    :param path: The path to the queue directory
    :param default: The name of the default yaml file
    :param filename: The name of the yaml file to save the parsed config to
    :return: The parsed config as a dictionar
    """
    # For convenience, we'll make a copy of the params
    params = dict(params)
    files = dict(params["files"])  
    # Setup the data 
    if "data" in params:
        if "data_file" not in files:
            data = dict(params["data"])
            assert data['filename'] is not None or data['path'] is not None or "data_file" in files, "Either path or filename must be specified for the data or data_file must be specified in the files section"
            data['path'] = params['data']['path'] if params['data']['path'] is not None else my_hash(data)
            data['filetype'] = params['data']['filetype'] if params['data']['filetype'] is not None else "npz"
            data['filename'] = params['data']['filename'] if params['data']['filename'] is not None else my_hash(data)
            files['data_file'] = str(Path(data['path'], data['filename'] + "." + data['filetype']).as_posix())
            del data['filename']
            del data['path']
            del data['filetype']
            params['data'] = data
    # Setup the model
    if "model" in params:
        model = dict(params["model"])
        if "model_file" not in files:
            assert model['filename'] is not None or model['path'] is not None , "Either path or filename must be specified for the model or model_file must be specified in the files section"
            model['path'] = params['model']['path'] if params['model']['path'] is not None else my_hash(params)
            model['filetype'] = params['model']['filetype'] if params['model']['filetype'] is not None else "pkl"
            model['filename'] = params['model']['filename'] if params['model']['filename'] is not None else my_hash(model)
            files['model_file'] = str(Path(model['path'], model['filename'] + "." + model['filetype']).as_posix())
            del model['path']
            del model['filename']
            del model['filetype']
            params['model'] = model
    # Setup the attack
    if "attack" in params:
        attack = dict(params["attack"])
        if "attack_file" not in files:
            assert "path" in attack  or "filename" in attack, "Either path or filename must be specified for the attack"
            attack['path'] = params['attack']['path'] if params['attack']['path'] is not None else my_hash(params)
            attack['filetype'] = params['attack']['filetype'] if params['attack']['filetype'] is not None else "pkl"
            attack['filename'] = params['attack']['filename'] if params['attack']['filename'] is not None else my_hash(attack) 
            files['attack_file'] = str(Path(attack['path'], attack['filename'] + "." + attack['filetype']).as_posix())
            del attack['path']
            del attack['filename']
            del attack['filetype']
            params['attack'] = attack
    # Setup the output
    params['files'] = files
    assert params['files']['path'] is None, f"Path is already specified as {params['files']['path']}"
    if stage is not None:
        params['files']['path'] = str(Path(stage, my_hash(params)).as_posix())
    elif stage is None:
        params['files']['path'] = str(Path(my_hash(params)).as_posix())
    params = DictConfig(params)
    params = OmegaConf.to_container(params, resolve=True)
    return params

def dump_params_to_queue(params:dict, stage) -> None:
    """
    Takes the params and dumps them to the path. Generates a unique filename based on the hash of the params if filename is None.
    :param params: The params to dump to the queue
    :param path: The path to the queue
    :param filename: The filename to save the params to
    :param return_path: Whether to return the path to the params
    :return: None
    """
    # Write the params to the queue
    filename = Path(params['files']['reports'], params['files']['path'], "params.yaml")
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, "w") as f:
        yaml.dump(params, f)
    logger.info(f"Wrote params to {filename}")
    return filename   
     
def dump_stage_to_queue(params:dict, stage:dict, stagename:str) -> None:
    """
    Takes the stage and dumps it to the path. Generates a unique filename based on the hash of the stage if filename is None.
    :param stage: The stage to dump to the queue
    :param path: The path to the queue
    :param filename: The filename to save the stage to
    :param return_path: Whether to return the path to the stage
    :return: None
    """
    # Write the stage to the queue

    filename = Path(params['files']['reports'], stagename, params['files']['path'], "dvc.yaml")
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(filename), "w") as f:
        yaml.dump(stage, f)
    assert Path(filename).exists(), f"Failed to write stage to {filename}"
    logger.info(f"Wrote stage to {filename}")
    return filename

            
def dump_params_to_default(params:dict, default:str = "params.yaml") -> None:
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
    return str(Path(default).as_posix())


def generate_stage_from_params(stage, params:dict, filename="dvc.yaml") -> None:
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
    if "outs" in pipe:
        new_outs = [Path(out).name for out in pipe['outs']]
        pipe['outs'] = new_outs
    if "metrics" in pipe:
        new_metrics = [Path(metric).name for metric in pipe['metrics']]
        pipe['metrics'] = new_metrics
    if "plots" in pipe:
        new_plots = [Path(plot).name for plot in pipe['plots']]
        pipe['plots'] = new_plots
    if "deps" in pipe:
        new_deps = [Path("..", "..", "..", dep).as_posix() for dep in pipe['deps']]
        pipe['deps'] = new_deps
    new_dict = {'stages' : {stage : pipe}}
    return new_dict
    
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
    import sys
    args = sys.argv
    if "--multirun" not in args:
        if "+default" not in args:
            if Path(os.getcwd(), "params.yaml").exists():
                Path(os.getcwd(), "params.yaml").unlink()
                args.append(f"+default={Path(os.getcwd(), 'params.yaml').as_posix()}")
    hydra_parser()
        
