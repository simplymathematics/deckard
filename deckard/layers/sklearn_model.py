import logging, argparse
from ..base import Data, Model, Experiment
import dvc.api
from os import path, mkdir
from typing import Union
from pathlib import Path
from .utils import make_output_folder, parse_config
logger = logging.getLogger(__name__)


def sklearn_model(args) -> Experiment:
    model = parse_config(args.config)
    if args.input_folder is None:
        assert Path(args.data_file).exists(), "Problem finding data file: {} in this working directory: {}".format(args.data_file, Path.cwd())
        data = Data(args.data_file)
    else:
        assert Path(args.input_folder, args.data_file).exists(), "Problem finding data file: {} in this working directory: {}".format(args.data_file, Path.cwd())
        data = Data(Path(args.input_folder, args.data_file))
    model = Model(model, art = False)
    exp = Experiment(data = data, model = model, filename = args.output_folder)
    exp(filename = args.output_name, path = args.output_folder)
    assert Path(args.output_folder, args.output_name).exists(), "Problem creating file: {}".format(Path(args.output_folder, args.output_name))
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
        if v is not None and not hasattr(args, k):
            setattr(args, k, v)
    output = make_output_folder(args.output_folder)
    assert Path(output).exists(), "Problem finding output folder: {}".format(output)
    assert isinstance(args.config, dict), "Config must be a dictionary. It is type: {}".format(type(args.config))
    assert isinstance(args.output_name, (str, Path)), "Output name must be a string. It is type: {}".format(type(args.output_name))
    assert isinstance(args.data_file, (str, Path)), "Data file must be a string. It is type: {}".format(type(args.data_file))
    sklearn_model(args)
   
