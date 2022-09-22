import logging, argparse
from pathlib import Path
from tqdm import tqdm
import dvc.api
from deckard.base import Scorer
from pandas import DataFrame

logger = logging.getLogger(__name__)


def evaluate(args) -> DataFrame:
    if args.inputs["recursive"] is True:
        ground_files = [
            path
            for path in Path(args.inputs["folder"]).rglob(
                "*" + args.inputs["ground_truth"],
            )
        ]
        predictions_files = [
            path
            for path in Path(args.inputs["folder"]).rglob(
                "*" + args.inputs["predictions"],
            )
        ]
    else:
        ground_files = [Path(args.inputs["folder"]) / args.inputs["ground_truth"]]
        predictions_files = [Path(args.inputs["folder"]) / args.inputs["predictions"]]
    is_regressor = (
        True if args.inputs["classifier"] is False else args.inputs["classifier"]
    )
    scorer = Scorer(is_regressor=is_regressor)
    big_dict = {}
    for gr, pr in tqdm(zip(ground_files, predictions_files), total=len(ground_files)):
        scores = scorer(gr, pr)
        parent = gr.parent
        big_dict[parent] = scores
    df = DataFrame.from_dict(big_dict, orient="index")
    Path(args.outputs["folder"]).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.outputs["folder"], args.metrics["scores"]))

    return df


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description="Run a preprocessor on a dataset")
    parser.add_argument(
        "--input_folder",
        "-i",
        type=str,
        default=".",
        help="Path to the model",
    )
    parser.add_argument(
        "--ground_truth_file",
        "-true",
        type=str,
        default="ground_truth.csv",
        help="Path to the model",
    )
    parser.add_argument(
        "--predictions_file",
        "-pred",
        type=str,
        default="predictions.csv",
        help="Path to the model",
    )
    parser.add_argument(
        "--layer_name",
        "-l",
        type=str,
        required=True,
        help='Name of layer, e.g. "attack"',
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        default=".",
        help="Path to the model",
    )
    parser.add_argument(
        "--output_file",
        "-f",
        type=str,
        default="scores.csv",
        help="Path to the model",
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

    evaluate(args)
