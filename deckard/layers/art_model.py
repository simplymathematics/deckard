import logging, argparse
from deckard.base import Data, Model, Experiment
import dvc.api
from pathlib import Path
from deckard.base.parse import make_output_folder, parse_config

logger = logging.getLogger(__name__)


def art_model(args) -> Experiment:
    assert Path(
        args.inputs["folder"],
        args.inputs["data"],
    ).exists(), "Problem finding data file: {} in this working directory: {}".format(
        args.inputs["data"],
        args.inputs["folder"],
    )
    assert (
        hasattr(args, "config") or "model" in args.inputs or "url" in args.inputs
    ), "Must have either a config file or a model"
    assert not (
        hasattr(args, "config") and "model" in args.inputs
    ), "Must have either a config, model file, or url, but only one."
    assert not (
        hasattr(args, "config") and "url" in args.inputs
    ), "Must have either a config, model file, or url, but only one."
    assert not (
        "url" in args.inputs and "model" in args.inputs
    ), "Must have either a config, model file, or url, but only one."
    if "model" in args.inputs:
        model = args.inputs["model"]
    elif "url" in args.inputs:
        model = args.inputs["url"]
    elif hasattr(args, "config"):
        model = parse_config(args.config)
    assert isinstance(
        model,
        object,
    ), "Problem parsing config file: {}. It is type: {}".format(
        args.config,
        type(args.config),
    )
    assert isinstance(
        args.outputs["model"],
        (str, Path),
    ), "Output name must be a string. It is type: {}".format(
        type(args.outputs["model"]),
    )
    assert isinstance(
        args.inputs["data"],
        (str, Path),
    ), "Data file must be a string. It is type: {}".format(type(args.inputs["data"]))
    data = Data(Path(args.inputs["folder"], args.inputs["data"]))
    if "url" in args.inputs:
        model = Model(
            args.outputs["model"],
            art=True,
            url=args.inputs["url"],
            model_type=args.inputs["type"],
            classifier=args.inputs["classifier"],
        )
    else:
        model = Model(
            art=True,
            model=model,
            model_type=args.inputs["type"],
            classifier=args.inputs["classifier"],
        )
    exp = Experiment(data=data, model=model)

    exp(model_file=args.outputs["model"], path=args.outputs["folder"])
    assert Path(
        args.outputs["folder"],
        args.outputs["model"],
    ).exists(), "Problem creating file: {}".format(
        Path(args.outputs["folder"], args.outputs["model"]),
    )
    logger.debug("Experiment hash: {}".format(hash(exp)))
    return exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a model on a dataset")
    parser.add_argument(
        "--input_folder",
        "-i",
        type=str,
        default=".",
        help="Path to the model",
    )
    parser.add_argument(
        "--output_folder",
        "-p",
        type=str,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--output_name",
        "-o",
        type=str,
        default=None,
        help="Name of the output file",
    )
    parser.add_argument(
        "--data_file",
        "-d",
        type=str,
        default="data.pkl",
        help="Path to the data file",
    )
    parser.add_argument(
        "--layer_name",
        "-l",
        type=str,
        required=True,
        help='Name of layer, e.g. "attack"',
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="Does Nothing")
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    assert isinstance(args.layer_name, str), "Layer name must be a string"
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and k in params:
            setattr(args, k, v)
    output = make_output_folder(args.outputs["folder"])
    assert Path(output).exists(), "Problem finding output folder: {}".format(output)
    art_model(args)
