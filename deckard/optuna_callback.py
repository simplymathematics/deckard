# Script to query the database

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import optuna
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf

storage = "sqlite:///optuna.db"
study_name = "gzip_knn_20-0"
metric_names = ["accuracy"]
directions = ["maximize"]
output_file = "optuna.csv"


@dataclass
class OptunaStudyDumpCallback(Callback):
    """
    Optuna callback to dump study results to CSV after multirun.

    Args:
        storage (str): Optuna storage URI.
        study_name (str): Name of the Optuna study.
        metric_names (Union[str, ListConfig, list]): Metric names to track.
        directions (Union[str, ListConfig, list]): Optimization directions.
        output_file (str): Path to output CSV file.
    """

    def __init__(
        self,
        storage: str,
        study_name: str,
        metric_names: Union[str, ListConfig, list],
        directions: Union[str, ListConfig, list],
        output_file: str,
    ):
        """
        Initialize the OptunaStudyDumpCallback.

        Args:
            storage (str): Optuna storage URI.
            study_name (str): Name of the Optuna study.
            metric_names (Union[str, ListConfig, list]): Metric names to track.
            directions (Union[str, ListConfig, list]): Optimization directions.
            output_file (str): Path to output CSV file.
        """
        self.storage = storage
        self.study_name = study_name
        # Make sure the folder exists
        db_file = self.storage.split("///")[-1]
        db_folder = Path(db_file).parent
        Path(db_folder).mkdir(parents=True, exist_ok=True)
        # Set metric names
        if isinstance(metric_names, ListConfig):
            self.metric_names = OmegaConf.to_container(
                metric_names,
                resolve=True,
            )
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
        """
        Called at the start of a multirun. Deletes existing study and creates a new one.

        Args:
            config (DictConfig): Hydra config.
            **kwargs: Additional keyword arguments.
        """
        try:
            study = optuna.load_study(self.study_name, storage=self.storage)
            study.delete_study(study_name=self.study_name, storage=self.storage)
        except Exception:
            pass
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

        if hasattr(study, "set_metric_names"):
            study.set_metric_names(self.metric_names)
        else:
            print("Cannot set metric names")

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        """
        Called at the end of a multirun. Saves the study trials to CSV.

        Args:
            config (DictConfig): Hydra config.
            **kwargs: Additional keyword arguments.
        """
        study = optuna.load_study(self.study_name, storage=self.storage)
        df = study.trials_dataframe()
        df.to_csv(self.output_file, index=False)
        print(f"Saved to {self.output_file}")
