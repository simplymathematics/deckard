
import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import dvc.api
from deckard.base import Data, Experiment, Model
from deckard.base.parse import (generate_experiment_list,
                                generate_object_from_tuple,
                                generate_tuple_from_yml)
from deckard.layers.utils import make_output_folder
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from .utils import make_output_folder, parse_config

logger = logging.getLogger(__name__)

def preprocess(input_folder:Union[str,Path], model:Union[str,Path], data:Union[str,Path], name:Union[str, Path], output_folder:Union[str,Path], preprocessor:object, position:int, model_type:str, layer:str) -> None:    
    data = Data(data)
    model_file = Path(input_folder, model)
    assert model_file.exists(), "Problem finding model file: {}".format(model_file)
    model = Model(model_file, art = False, model_type = model_type)
    exp = Experiment(data = data, model = model)
    assert isinstance(exp, Experiment), "Problem initializing experiment"
    new = deepcopy(exp)
    new.insert_sklearn_preprocessor(preprocessor = preprocessor, position = position, name = layer)
    assert isinstance(new, Experiment), "Problem inserting preprocessor"
    assert isinstance(new.model.model, Pipeline), "Problem inserting preprocessor. Model is not a Pipeline. It is a {}".format(type(new.model))
    new(filename = name, path = output_folder)
    assert Path(output_folder, name).exists(), "Problem creating file: {}".format(Path(output_folder, name))
    return new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a preprocessor on a dataset')
    parser.add_argument('--input_folder', '-i', type=str, default = ".", help='Path to the model')
    parser.add_argument('--output_folder', '-p', type=str, help='Path to the output folder')
    parser.add_argument('--output_name','-o', type=str, default=None, help='Name of the output file')
    parser.add_argument('--data_file', '-d', type=str, default = "data.pkl", help='Path to the data file')
    parser.add_argument('--layer_name','-l', type=str, required = True, help='Name of layer, e.g. "attack"')
    parser.add_argument('--config','-c', type=str, default = None, help='Path to the attack config file')
    parser.add_argument('--position', '-n', type=int, default = 0, help='Position of the preprocessor in the pipeline')
    parser.add_argument('--input_name', '-m', type=str, default = None, help='Name of the input file')
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    assert isinstance(args.layer_name, str), "Layer name must be a string. It is a {}".format(type(args.layer_name))
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    # create output folder
    assert isinstance(args.output_folder, (str, Path)), "Output folder must be a string or a Path object. It is a {}".format(type(args.output_folder))
    output_folder = make_output_folder(args.output_folder)
    assert isinstance(args.config, dict), "Config must be a dictionary. It is a {}".format(type(args.config))
    preprocessor = parse_config(args.config)
    assert isinstance(args.input_name, (str, Path)), "Input name must be a string or a Path object. It is a {}".format(type(args.input_name))
    assert isinstance(args.data_file, (str, Path)), "Data file must be a string or a Path object. It is a {}".format(type(args.data_file))
    assert isinstance(args.output_name, (str, Path)), "Output name must be a string or a Path object. It is a {}".format(type(args.output_name))
    assert isinstance(args.position, int), "Position must be an integer. It is a {}".format(type(args.position))
    assert isinstance(preprocessor, object), "Preprocessor must be an object. It is a {}".format(type(preprocessor))
    assert isinstance(args.input_folder, (str, Path)), "Input folder must be a string or a Path object. It is a {}".format(type(args.input_folder))
    preprocess(input_folder = args.input_folder, model = args.input_name, data = args.data_file, name = args.output_name, position = args.position, output_folder = output_folder, preprocessor = preprocessor, model_type = args.model_type, layer = args.layer_name)

