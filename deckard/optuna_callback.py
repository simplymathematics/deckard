# Script to query the database

from omegaconf import DictConfig, ListConfig, OmegaConf
from dataclasses import dataclass
import optuna
from hydra.experimental.callback import Callback
from typing import Union
from pathlib import Path



logger = optuna.logging.get_logger(__name__)

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
        # delete the study if it exists
        try:
            study = optuna.load_study(self.study_name, storage=self.storage)
            study.delete_study(study_name=self.study_name, storage=self.storage)
        except:  # noqa E722
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
            logger.warning("Optuna version does not support set_metric_names. Please upgrade to Optuna 3.0.0 or higher.")

    def on_multirun_end(self, config: DictConfig, **kwargs) -> None:
        study = optuna.load_study(self.study_name, storage=self.storage)
        df = study.trials_dataframe()
        df.to_csv(self.output_file, index=False)
        print(f"Saved to {self.output_file}")
