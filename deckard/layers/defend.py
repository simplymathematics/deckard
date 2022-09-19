import logging
import os
from pathlib import Path

import numpy as np
from deckard.base import Data, Experiment, Model

logger = logging.getLogger(__name__)


def defend(args) -> None:
    data = Data(args.inputs["data"], train_size=100)
    data()
    mini = np.amin(data.X_train)
    maxi = np.amax(data.X_train)
    clip_values = (mini, maxi)
    model_file = Path(args.inputs["folder"], args.inputs["model"])
    model = Model(model_file, art=True, clip_values=clip_values, defence = args.config, model_type = args.inputs['type'])
    model(art = True)
    experiment = Experiment(data=data, model=model, is_fitted=True)
    filename = experiment(path=args.outputs["folder"], model_file=args.outputs["model"])
    return filename


if __name__ == "__main__":
    # args
    import argparse

    import dvc.api

    parser = argparse.ArgumentParser(
        description="Prepare model and dataset as an experiment object. Then runs the experiment."
    )
    parser.add_argument(
        "--layer_name",
        "-l",
        type=str,
        required=True,
        help='Name of layer, e.g. "attack"',
    )
    parser.add_argument(
        "--input_model", "-m", type=str, default=None, help="Name of the model"
    )
    parser.add_argument(
        "--input_folder", "-i", type=str, default=None, help="Path to the model"
    )
    parser.add_argument(
        "--model_type", "-t", type=str, default=None, help="Type of the model"
    )
    parser.add_argument(
        "--output_folder",
        "-p",
        type=str,
        default=None,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--output_name", "-o", type=str, default=None, help="Name of the output file"
    )
    parser.add_argument(
        "--data_file", "-d", type=str, default=None, help="Path to the data file"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to the attack config file"
    )
    cli_args = parser.parse_args()
    params = dvc.api.params_show()[cli_args.layer_name]
    args = argparse.Namespace(**params)
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    if not os.path.exists(args.outputs["folder"]):
        logger.warning(
            "Model path {} does not exist. Creating it.".format(args.outputs["folder"])
        )
        os.mkdir(args.outputs["folder"])
    ART_DATA_PATH = args.outputs["folder"]
    defend(args)
