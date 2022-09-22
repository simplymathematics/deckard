import argparse
import logging
from pathlib import Path
from typing import Callable, Union
import dvc.api
import numpy as np
from deckard.base import Data, Model

# from yellowbrick.exceptions import YellowbrickValueError

logger = logging.getLogger(__name__)


def visualise_sklearn_classifier_experiment(
    args: argparse.Namespace,
    path: Union[str, Path] = Path("."),
    type: str = "ROC_AUC",
) -> None:
    """
    Visualise the results of a single experiment.
    :param args: a dictionary read at run-time
    :param path: the path to the experiment
    :param type: the type of visualisation to perform
    """
    assert isinstance(args, argparse.Namespace), "args must be a dictionary-like object"
    assert isinstance(path, (str, Path)), "path must be a string or a Path object"
    assert isinstance(type, str), "type must be a string"

    data = Data(Path(args.inputs["folder"], args.inputs["data"]))
    data()
    model = Model(
        Path(args.inputs["folder"], args.inputs["model"]),
        model_type=args.inputs["type"],
        art=args.inputs["art"],
    )
    model()
    try:
        classes = list(set(data.y_train))
        y_train = data.y_train
        y_test = data.y_test
    except:  # noqa: E722
        y_train = [np.argmax(y) for y in data.y_train]
        y_test = [np.argmax(y) for y in data.y_test]
        classes = list(set(y_train))
    if hasattr(args, "art") and args.art is True:
        logger.info("Using ART model")
        viz_mod = model.model
    else:
        viz_mod = model
    assert isinstance(
        viz_mod,
        (Callable, Model),
    ), "model must be a callable object. It is type {}".format(type(viz_mod))
    if type == "ROC_AUC":
        from yellowbrick.classifier import ROCAUC

        func = ROCAUC(viz_mod.model, classes=classes, force_model=True)
        outpath = Path(args.outputs["folder"], "ROC_AUC.pdf")
        outpath = outpath.resolve()
        outpath.parent.mkdir(parents=True, exist_ok=True)
    else:
        raise NotImplementedError("Visualisation type {} not implemented".format(type))
    func.fit(data.X_train, y_train)
    func.score(data.X_test, y_test)
    func.show(outpath=outpath)
    logger.info("Saving visualisation to {}".format(outpath))
    return outpath.resolve()


if __name__ == "__main__":
    # Initialize logger
    # command line arguments
    parser = argparse.ArgumentParser(description="Run a preprocessor on a dataset")
    parser.add_argument(
        "--layer_name",
        "-l",
        type=str,
        default="visualise",
        help='Name of layer, e.g. "attack"',
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to the attack config file",
    )
    args = parser.parse_args()
    # parse arguments
    cli_args = parser.parse_args()
    params = dvc.api.params_show()
    args = argparse.Namespace(**params[cli_args.layer_name])
    for k, v in vars(cli_args).items():
        if v is not None and not hasattr(args, k):
            setattr(args, k, v)
    logger.info(f"Running {cli_args.layer_name} with args: {args}")
    # assert Path(args.config).exists(), f"Config file {args.config} does not exist"
    output = visualise_sklearn_classifier_experiment(args, path=args.inputs["folder"])
