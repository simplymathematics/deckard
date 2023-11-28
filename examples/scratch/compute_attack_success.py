import logging
import argparse
from pathlib import Path
import pandas as pd
import json

logger = logging.getLogger(__name__)


def read_data_file(file: str, target=None):
    filetype = Path(file).suffix
    if filetype == ".csv":
        data = pd.read_csv(file)
    elif filetype == ".parquet":
        data = pd.read_parquet(file)
    elif filetype == ".json":
        with open(file, "r") as f:
            data = json.load(f)
        data = pd.DataFrame(data, index=range(len(data)))
    else:
        raise ValueError(f"Unknown file type: {filetype}")
    # to numpy
    data = data.to_numpy()
    logger.info(f"Loaded data from {file} with shape {data.shape}")
    return data


def write_data_file(data, file: str):
    filetype = Path(file).suffix
    if filetype == ".csv":
        old = pd.read_csv(file)
        data = old.to_dict()
        data.update(**data)
        data = pd.DataFrame(data)
        data.to_csv(file, index=False)
    elif filetype == ".json":
        with open(file, "r") as f:
            old = json.load(f)
        old.update(**data)
        with open(file, "w") as f:
            json.dump(old, f)
    else:
        raise ValueError(f"Unknown file type: {filetype}")
    return None


if __name__ == "__main__":
    attack_success_parser = argparse.ArgumentParser(
        description="Compute attack success",
    )
    attack_success_parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    attack_success_parser.add_argument(
        "-b",
        "--ben_predictions_file",
        help="Full path to the predictions file",
        required=True,
    )
    attack_success_parser.add_argument(
        "-a",
        "--adv_predictions_file",
        help="Full path to the labels file",
        required=True,
    )
    attack_success_parser.add_argument(
        "-l",
        "--labels_file",
        help="Full path to the predictions file",
        required=True,
    )
    attack_success_parser.add_argument(
        "-i",
        "--input_score_file",
        default=None,
        required=True,
    )
    attack_success_parser.add_argument(
        "-o",
        "--output_score_file",
        help="Full path to the labels file",
        required=True,
    )
    args = attack_success_parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # labels = read_data_file(args.labels_file)
    # nb_classes = len(np.unique(labels))
    # ben_pred = read_data_file(args.ben_predictions_file)
    # adv_pred = read_data_file(args.adv_predictions_file)
    # adv_pred = to_categorical(labels=adv_pred, nb_classes = nb_classes)
    # ben_acc, _ = compute_accuracy(ben_pred, labels[:len(ben_pred)])
    # adv_acc, _ = compute_accuracy(adv_pred, labels[:len(adv_pred)])
    # adv_suc, _ = compute_accuracy(adv_pred, ben_pred[:len(adv_pred)])
    # adv_suc = 1 - adv_suc
    # logger.info(f"Benign accuracy: {ben_acc}")
    # logger.info(f"Adversarial accuracy: {adv_acc}")
    # logger.info(f"Adversarial success: {adv_suc}")
    # score_dict = {"ben_acc": ben_acc, "adv_acc": adv_acc, "adv_suc": adv_suc}
    # with open(args.input_score_file, "r") as f:
    #     old = json.load(f)
    # new = old.copy()
    # new.update(**score_dict)
    # with open(args.output_score_file, "w") as f:
    #     json.dump(new, f)
