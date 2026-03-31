# Script to query the database

from omegaconf import DictConfig, ListConfig, OmegaConf
from dataclasses import dataclass
import optuna
from hydra.experimental.callback import Callback
import argparse
from typing import Union
from pathlib import Path
import logging

storage = "sqlite:///optuna.db"
study_name = "gzip_knn_20-0"
metric_names = ["accuracy"]
directions = ["maximize"]
output_file = "optuna.csv"

logger = logging.getLogger(__name__)


@dataclass
class OptunaStudyDumpCallback(Callback):
    def __init__(
        self,
        storage: str,
        study_name: str,
        metric_names: Union[str, ListConfig, list],
        directions: Union[str, ListConfig, list],
        output_file: str,
    ):
        self.storage = storage
        self.study_name = study_name
        # Make sure the folder exists
        db_file = self.storage.split("///")[-1]
        db_folder = Path(db_file).parent
        Path(db_folder).mkdir(parents=True, exist_ok=True)
        # Set metric names
        if isinstance(metric_names, ListConfig):
            self.metric_names = OmegaConf.to_container(metric_names, resolve=True)
        elif isinstance(metric_names, list):
            self.metric_names = metric_names
        else:
            self.metric_names = [metric_names]
        # Set direction
        if isinstance(directions, ListConfig):
            self.directions = OmegaConf.to_container(directions, resolve=True)
        elif isinstance(directions, list):
            self.directions = directions
        else:
            self.directions = [directions]
        self.output_file = output_file
        super().__init__()

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        study = self.get_study()
        if hasattr(study, "set_metric_names"):
            study.set_metric_names(self.metric_names)
        else:
            logger.info("Cannot set metric names")

    def on_run_start(self, config: DictConfig, **kwargs: optuna.Any) -> None:
        study = self.get_study()
        if hasattr(study, "set_metric_names"):
            study.set_metric_names(self.metric_names)
        else:
            logger.info("Cannot set metric names")

    def get_study(self):
        if len(self.directions) == 1:
            direction = self.directions[0]
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=direction,
                load_if_exists=True,
            )
        else:
            directions = self.directions
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                directions=directions,
                load_if_exists=True,
            )
        return study

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        study = optuna.load_study(self.study_name, storage=self.storage)
        df = study.trials_dataframe()
        df.to_csv(self.output_file, index=False)
        logger.info(f"Saved to {self.output_file}")


def multirun_call(args):
    storage = args.storage
    study_name = args.study_name
    metric_names = (
        args.metric_names
        if isinstance(args.metric_names, list)
        else [args.metric_names]
    )
    directions = (
        args.directions if isinstance(args.directions, list) else [args.directions]
    )
    output_file = args.output_file
    if output_file is not None:
        callback = OptunaStudyDumpCallback(
            storage,
            study_name,
            metric_names,
            directions,
            output_file,
        )
        callback.on_multirun_start()


optuna_callback_parser = argparse.ArgumentParser()
optuna_callback_parser.add_argument("--storage", type=str, default=storage)
optuna_callback_parser.add_argument("--study_name", type=str, default=study_name)
optuna_callback_parser.add_argument(
    "--metric_names",
    type=str,
    nargs="+",
    default=metric_names,
)
optuna_callback_parser.add_argument(
    "--directions",
    type=str,
    nargs="+",
    default=directions,
)
optuna_callback_parser.add_argument("--output_file", type=str, default=output_file)

if __name__ == "__main__":
    args = optuna_callback_parser.parse_args()
    multirun_call(args)
