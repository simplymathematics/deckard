import logging, argparse
from deckard.base import Data, Model, Experiment
import dvc.api
from os import path, mkdir
from typing import Union
from pathlib import Path
from deckard.layers.utils import make_output_folder, parse_config
logger = logging.getLogger(__name__)


def art_model(args) -> Experiment:
    assert Path(args.input_folder, args.data_file).exists(), "Problem finding data file: {} in this working directory: {}".format(args.data_file, args.input_folder)
    data = Data(Path(args.input_folder, args.data_file))
    model = parse_config(args.config)
    model = Model(model, art = True)
    exp = Experiment(data = data, model = model, filename = args.output_folder)
    exp(filename = args.output_name, path = args.output_folder)
    assert Path(args.output_folder, args.output_name).exists(), "Problem creating file: {}".format(Path(args.output_folder, args.output_name))
    logger.debug("Experiment hash: {}".format(hash(exp)))
    return exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a model on a dataset')
    parser.add_argument('--input_folder', '-i', type=str, default = ".", help='Path to the model')
    parser.add_argument('--output_folder', '-p', type=str, help='Path to the output folder')
    parser.add_argument('--output_name','-o', type=str, default=None, help='Name of the output file')
    parser.add_argument('--data_file', '-d', type=str, default = "data.pkl", help='Path to the data file')
    parser.add_argument('--layer_name','-l', type=str, required = True, help='Name of layer, e.g. "attack"')
    parser.add_argument('--config','-c', type=str, default = None, help='Does Nothing')
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
    assert hasattr(args, "config") or hasattr(args, "input_model") or hasattr(args, 'url'), "Must have either a config file or a model"
    assert not (hasattr(args, "config") and hasattr(args, "input_model")), "Must have either a config, model file, or url, but only one."
    assert not (hasattr(args, "config") and hasattr(args, 'url')), "Must have either a config, model file, or url, but only one."
    assert not (hasattr(args, "input_model") and hasattr(args, 'url')), "Must have either a config, model file, or url, but only one."
    if hasattr(args, "config"):
        model = parse_config(args.config)
    elif hasattr(args, "input_model"):
        model = args.input_model
    elif hasattr(args, 'url'):
        model = args.url
    assert isinstance(model, object), "Problem parsing config file: {}. It is type: {}".format(args.config, type(args.config))
    assert isinstance(args.output_name, (str, Path)), "Output name must be a string. It is type: {}".format(type(args.output_name))
    assert isinstance(args.data_file, (str, Path)), "Data file must be a string. It is type: {}".format(type(args.data_file))
    art_model(args)