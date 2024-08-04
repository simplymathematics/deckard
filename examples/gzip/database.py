# Script to query the database

from omegaconf import DictConfig, ListConfig, OmegaConf
from dataclasses import dataclass
import optuna
from pathlib import Path
from hydra.experimental.callback import Callback
import argparse
from typing import Union
storage = "sqlite:///optuna.db"
study_name = "gzip_knn_20-0"
metric_names = ["accuracy"]
directions = ["maximize"]
output_file = "optuna.csv"

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
        if isinstance(metric_names, ListConfig):
            self.metric_names = OmegaConf.to_container(metric_names, resolve=True)
        elif isinstance(metric_names, list):
            self.metric_names = metric_names
        else:
            self.metric_names = [metric_names]
        if isinstance(directions, ListConfig):
            self.metric_names = OmegaConf.to_container(directions, resolve=True)
        elif isinstance(directions, list):
            self.directions = directions
        else:
            self.directions = [directions]
        self.output_file = output_file
        super().__init__()

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        studies = optuna.get_all_study_names(self.storage)
        study_names = [study for study in studies]
        # study_names = [study.study_name for study in studies]
        assert (
            self.study_name in study_names
        ), f"Study {self.study_name} not found in {study_names}"
        study = optuna.load_study(self.study_name, storage=self.storage)
        if hasattr(study, "set_metric_names"):
            study.set_metric_names(self.metric_names)
        else:
            print("Cannot set metric names")

    def on_multirun_end(self, *args, **kwargs) -> None:
        studies = optuna.get_all_study_names(self.storage)
        study_names = [study for study in studies]
        assert (
            self.study_name in study_names
        ), f"Study {self.study_name} not found in {study_names}"
        study = optuna.load_study(self.study_name, storage=self.storage)
        df = study.trials_dataframe()
        if len(self.metric_names) == 1:
            metric_names = [f"value_{metric}" for metric in self.metric_names]
            df = df.sort_values(metric_names[0], ascending=False)
        else:
            metric_names = [f"values_{metric}" for metric in self.metric_names]
            df = df.sort_values(metric_names, ascending=False)
        suffix = Path(self.output_file).suffix
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        if suffix in [".csv"]:
            df.to_csv(self.output_file, index=False)
        elif suffix in [".json"]:
            df.to_json(self.output_file, orient="records")
        else:
            raise ValueError(f"Unknown file type {suffix}")

        assert Path(self.output_file).exists(), f"File {self.output_file} not found"


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

    callback = OptunaStudyDumpCallback(
        storage,
        study_name,
        metric_names,
        directions,
        output_file,
    )
    callback.on_multirun_start()
    callback.on_multirun_end()


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
