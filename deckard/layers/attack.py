import argparse
import logging
import os
from pathlib import Path

import dvc.api
import numpy as np
from deckard.base import AttackExperiment, Data, Model

logger = logging.getLogger(__name__)


# def __init__(
#         self,
#         data : Data,
#         model: Model,
#         model_type: str = "sklearn",
#         defence:dict = None,
#         pipeline:dict = None,
#         path=".",
#         is_fitted: bool = False,
#         classifier=True,
#         art: bool = True,
#         fit_params: dict = None,
#         predict_params: dict = None,
#         clip_values: tuple = None,
# ):


def attack_layer(args) -> None:
    data = Data(Path(args.inputs["folder"], args.inputs["data"]))
    data()
    if "attack_size" in args.inputs:
        data.X_test = data.X_test[: args.inputs["attack_size"]]
        data.y_test = data.y_test[: args.inputs["attack_size"]]
    mini = np.amin(data.X_train)
    maxi = np.amax(data.X_train)
    _ = (mini, maxi)
    model_file = Path(args.inputs["folder"], args.inputs["model"])
    art_model = Model(model_file, model_type=args.inputs["type"], art=True)
    art_model(art=True)

    experiment = AttackExperiment(
        data=data,
        model=art_model,
        attack=args.config,
        is_fitted=args.inputs["is_fitted"] if "is_fitted" in args.inputs else False,
        fit_params=args.inputs["fit_params"] if "fit_params" in args.inputs else None,
        predict_params=args.inputs["predict_params"]
        if "predict_params" in args.inputs
        else None,
    )
    experiment(path=args.outputs["folder"], model_file=args.outputs["model"])
    return None


if __name__ == "__main__":
    # args

    parser = argparse.ArgumentParser(
        description="Prepare model and dataset as an experiment object. Then runs the experiment.",
    )
    parser.add_argument(
        "--layer_name",
        "-l",
        type=str,
        required=True,
        help='Name of layer, e.g. "attack"',
    )
    cli_args = parser.parse_args()
    params = dvc.api.params_show()[cli_args.layer_name]
    args = argparse.Namespace(**params)
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    if not os.path.exists(args.outputs["folder"]):
        logger.warning(
            "Model path {} does not exist. Creating it.".format(args.outputs["folder"]),
        )
        os.mkdir(args.outputs["folder"])
    ART_DATA_PATH = args.outputs["folder"]
    attack_layer(args)
