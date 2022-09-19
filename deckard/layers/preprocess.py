import argparse
import logging
import os
from copy import deepcopy
from pathlib import Path

import dvc.api
from deckard.base import Data, Experiment, Model
from deckard.layers.utils import make_output_folder
from sklearn.pipeline import Pipeline
from deckard.layers.utils import make_output_folder, parse_config

logger = logging.getLogger(__name__)


def preprocess(args) -> Experiment:
    data = Data(args.inputs["data"])
    model_file = Path(args.inputs["folder"], args.inputs["model"])
    preprocessor = parse_config(args.config)
    assert model_file.exists(), "Problem finding model file: {}".format(model_file)
    model = Model(model_file, art=False, model_type=args.inputs["type"])
    model()
    exp = Experiment(data=data, model=model)
    assert isinstance(exp, Experiment), "Problem initializing experiment"
    new = deepcopy(exp)
    new.model.insert_sklearn_preprocessor(
        preprocessor=preprocessor, position=args.position, name=args.layer_name
    )
    assert isinstance(new, Experiment), "Problem inserting preprocessor"
    assert isinstance(
        new.model.model, Pipeline
    ), "Problem inserting preprocessor. Model is not a Pipeline. It is a {}".format(
        type(new.model)
    )
    new(model_file=args.outputs["model"], path=output_folder)
    assert Path(
        output_folder, args.outputs["model"]
    ).exists(), "Problem creating file: {}".format(
        Path(output_folder, args.outputs["model"])
    )
    logger.debug("Preprocessing complete")
    return new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a preprocessor on a dataset")
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
        "--config", "-c", type=str, default=None, help="Path to the attack config file"
    )
    parser.add_argument(
        "--position",
        "-n",
        type=int,
        default=0,
        help="Position of the preprocessor in the pipeline",
    )
    parser.add_argument(
        "--input_name", "-m", type=str, default=None, help="Name of the input file"
    )
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    assert isinstance(
        args.layer_name, str
    ), "Layer name must be a string. It is a {}".format(type(args.layer_name))
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and not hasattr(args, k):
            setattr(args, k, v)
    # create output folder
    assert isinstance(
        args.outputs["folder"], (str, Path)
    ), "Output folder must be a string or a Path object. It is a {}".format(
        type(args.outputs["folder"])
    )
    output_folder = make_output_folder(args.outputs["folder"])
    assert isinstance(
        args.config, dict
    ), "Config must be a dictionary. It is a {}".format(type(args.config))
    assert isinstance(
        args.inputs["data"], (str, Path)
    ), "Input name must be a string or a Path object. It is a {}".format(
        type(args.inputs["file"])
    )
    assert isinstance(
        args.inputs["model"], (str, Path)
    ), "Input name must be a string or a Path object. It is a {}".format(
        type(args.inputs["file"])
    )
    assert isinstance(
        args.inputs["data"], (str, Path)
    ), "Data file must be a string or a Path object. It is a {}".format(
        type(args.inputs["data"])
    )
    assert isinstance(
        args.outputs["model"], (str, Path)
    ), "Output name must be a string or a Path object. It is a {}".format(
        type(args.outputs["model"])
    )
    assert isinstance(
        args.position, int
    ), "Position must be an integer. It is a {}".format(type(args.position))
    assert isinstance(
        args.inputs["folder"], (str, Path)
    ), "Input folder must be a string or a Path object. It is a {}".format(
        type(args.inputs["folder"])
    )
    preprocess(args)
