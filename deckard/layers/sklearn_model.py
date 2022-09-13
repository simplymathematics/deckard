import logging, argparse
from ..base import Data, Model, Experiment
from ..base.parse import generate_object_from_tuple, generate_tuple_from_yml, generate_experiment_list
import dvc.api
from os import path, mkdir
from typing import Union
from pathlib import Path
from .utils import make_output_folder, parse_config
logger = logging.getLogger(__name__)


def sklearn_model(output_name:Union[str, Path], output_folder:Union[str, Path], data_file:Union[str, Path], model:object, input_folder:Union[str, Path] = None) -> Experiment:
    if input_folder is None:
        assert Path(data_file).exists(), "Problem finding data file: {} in this working directory: {}".format(data_file, Path.cwd())
        data = Data(data_file)
    else:
        assert Path(input_folder, data_file).exists(), "Problem finding data file: {} in this working directory: {}".format(data_file, Path.cwd())
        data = Data(Path(input_folder, data_file))
    model = Model(model, art = False)
    exp = Experiment(data = data, model = model, filename = output_folder)
    exp(filename = output_name, path = output_folder)
    assert Path(output_folder, output_name).exists(), "Problem creating file: {}".format(Path(output_folder, output_name))
    return exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a model on a dataset')
    parser.add_argument('--input_folder', '-i', type=str, default = ".", help='Path to the model')
    parser.add_argument('--output_folder', '-p', type=str, help='Path to the output folder')
    parser.add_argument('--output_name','-o', type=str, default=None, help='Name of the output file')
    parser.add_argument('--data_file', '-d', type=str, default = "data.pkl", help='Path to the data file')
    parser.add_argument('--layer_name','-l', type=str, required = True, help='Name of layer, e.g. "attack"')
    parser.add_argument('--config','-c', type=str, default = None, help='Control Model Config')
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    assert isinstance(args.layer_name, str), "Layer name must be a string"
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    output = make_output_folder(args.output_folder)
    assert Path(output).exists(), "Problem finding output folder: {}".format(output)
    assert isinstance(args.config, dict), "Config must be a dictionary. It is type: {}".format(type(config))
    model = parse_config(args.config)
    assert isinstance(model, object), "Problem parsing config file: {}. It is type: {}".format(args.config, type(config))
    assert isinstance(args.output_name, (str, Path)), "Output name must be a string. It is type: {}".format(type(args.output_name))
    assert isinstance(args.data_file, (str, Path)), "Data file must be a string. It is type: {}".format(type(args.data_file))
    sklearn_model(input_folder = args.input_folder, output_folder = output, output_name = args.output_name, data_file = args.data_file, model = model)
   
