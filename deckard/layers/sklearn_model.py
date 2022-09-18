import logging, argparse
from deckard.base import Data, Model, Experiment
import dvc.api
from os import path, mkdir
from typing import Union
from pathlib import Path
from deckard.layers.utils import make_output_folder, parse_config

logger = logging.getLogger(__name__)


def sklearn_model(args) -> Experiment:
    model = parse_config(args.config)
    if args.inputs["folder"] is None:
        assert Path(
            args.inputs["data"]
        ).exists(), "Problem finding data file: {} in this working directory: {}".format(
            args.inputs["data"], Path.cwd()
        )
        data = Data(args.inputs["data"])
    else:
        assert Path(
            args.inputs["folder"], args.inputs["data"]
        ).exists(), "Problem finding data file: {} in this working directory: {}".format(
            args.inputs["data"], Path.cwd()
        )
        data = Data(Path(args.inputs["folder"], args.inputs["data"]))
    model = Model(model, art=False)
    exp = Experiment(data=data, model=model)
    exp(filename=args.outputs["model"], path=args.outputs["folder"])
    assert Path(
        args.outputs["folder"], args.outputs["model"]
    ).exists(), "Problem creating file: {}".format(
        Path(args.outputs["folder"], args.outputs["model"])
    )
    return exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model on a dataset")
    parser.add_argument(
        "--input_folder", "-i", type=str, default=".", help="Path to the model"
    )
    parser.add_argument(
        "--output_folder", "-p", type=str, help="Path to the output folder"
    )
    parser.add_argument(
        "--output_name", "-o", type=str, default=None, help="Name of the output file"
    )
    parser.add_argument(
        "--data_file", "-d", type=str, default="data.pkl", help="Path to the data file"
    )
    parser.add_argument(
        "--layer_name",
        "-l",
        type=str,
        required=True,
        help='Name of layer, e.g. "attack"',
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Control Model Config"
    )
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    assert isinstance(args.layer_name, str), "Layer name must be a string"
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and not hasattr(args, k):
            setattr(args, k, v)
    output = make_output_folder(args.outputs["folder"])
    assert Path(output).exists(), "Problem finding output folder: {}".format(output)
    assert isinstance(
        args.config, dict
    ), "Config must be a dictionary. It is type: {}".format(type(args.config))
    assert isinstance(
        args.outputs["model"], (str, Path)
    ), "Output name must be a string. It is type: {}".format(
        type(args.outputs["model"])
    )
    assert isinstance(
        args.inputs["data"], (str, Path)
    ), "Data file must be a string. It is type: {}".format(type(args.inputs["data"]))
    sklearn_model(args)
