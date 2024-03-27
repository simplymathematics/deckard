import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union
from copy import deepcopy
from omegaconf import OmegaConf
from ..utils import my_hash

logger = logging.getLogger(__name__)

__all__ = ["FileConfig"]


@dataclass
class FileConfig:
    reports: Union[str, None] = "reports"
    data_dir: Union[str, None] = "data"
    model_dir: Union[str, None] = "models"
    attack_dir: Union[str, None] = "attacks"
    name: Union[str, None] = None
    stage: Union[str, None] = None
    files: dict = field(default_factory=dict)

    def __init__(
        self,
        reports: str = "reports",
        data_dir: str = "data",
        model_dir: str = "models",
        attack_dir="attacks",
        attack_type: str = ".pkl",
        data_type: str = ".pkl",
        model_type: str = ".pkl",
        directory=None,
        stage=None,
        name=None,
        files={},
        **kwargs,
    ):
        """
        A class to manage the file system for an experiment.
        :param reports: The directory to store the reports. Default is "reports". If stage is not None, the reports will be stored in reports/stage/name.
        :param data_dir: The directory to store the data. Default is "data".
        :param model_dir: The directory to store the models. Default is "models".
        :param attack_dir: The directory to store the attacks. Default is "attacks".
        :param stage: The stage of the experiment. Default is None.
        :param name: The name of the experiment. Default is None.
        :param files: A dictionary of the files to be used. Default is an empty dictionary.
        :param directory: The directory to store the files. Default is None. If not None, all files will be stored in this directory.
        :param kwargs: Additional keyword arguments to be added to the files dictionary.
        :return: A FileConfig object.
        """
        self._target_ = "deckard.base.files.FileConfig"
        files = OmegaConf.merge(files, kwargs)
        self.reports = str(Path(reports).as_posix()) if reports is not None else None
        self.data_dir = str(Path(data_dir).as_posix()) if data_dir is not None else None
        self.model_dir = (
            str(Path(model_dir).as_posix()) if model_dir is not None else None
        )
        self.attack_dir = (
            str(Path(attack_dir).as_posix()) if attack_dir is not None else None
        )
        self.data_type = data_type if data_type else None
        self.model_type = model_type if model_type else None
        self.attack_type = attack_type if attack_type else None
        self.directory = (
            (
                Path(directory).as_posix()
                if Path(directory).is_absolute()
                else Path(Path(), directory).as_posix()
            )
            if directory
            else None
        )
        self.stage = stage if stage else None
        self.files = files if files else {}
        logger.debug(f"FileConfig init: {self.files}")
        if name is None:
            self.name = my_hash(self)
        else:
            self.name = name

    def __call__(self):
        files = dict(self.get_filenames())
        return files

    def get_filenames(self):
        files = deepcopy(self.files)
        files = self._set_filenames(**files)
        return files

    def _set_filenames(self, **kwargs):
        name = self.name
        stage = self.stage
        if hasattr(self, "files"):
            kwargs.update(self.files)
        files = dict(kwargs)
        new_files = {}
        directory = self.directory
        reports = self.reports
        data_dir = self.data_dir
        model_dir = self.model_dir
        attack_dir = self.attack_dir
        data_type = self.data_type
        model_type = self.model_type
        attack_type = self.attack_type
        reports = (
            str(Path(directory, reports).as_posix()) if reports is not None else None
        )
        data_dir = (
            str(Path(directory, data_dir).as_posix()) if data_dir is not None else None
        )
        model_dir = (
            str(Path(directory, model_dir).as_posix())
            if model_dir is not None
            else None
        )
        attack_dir = (
            str(Path(directory, attack_dir).as_posix())
            if attack_dir is not None
            else None
        )
        if name is None and stage is None:
            path = Path(reports)
        elif name is not None and stage is None:
            path = Path(reports, name)
        elif name is None and stage is not None:
            path = Path(reports, stage)
        else:
            path = Path(reports, stage, name)
        for kwarg in files:
            name = files.get(kwarg)
            if "data_file" == kwarg and data_dir is not None:
                new_path = Path(data_dir, name)
                if new_path.suffix != data_type:
                    new_path = Path(data_dir, Path(name).stem + data_type)
                new_files[kwarg] = str(new_path.as_posix())
            elif "model_file" == kwarg and model_dir is not None:
                new_path = Path(model_dir, name)
                if new_path.suffix != model_type:
                    new_path = Path(model_dir, Path(name).stem + model_type)
                new_files[kwarg] = str(new_path.as_posix())
            elif "attack_file" == kwarg and attack_dir is not None:
                new_path = Path(attack_dir, name)
                if new_path.suffix != attack_type:
                    new_path = Path(attack_dir, Path(name).stem + attack_type)
                new_files[kwarg] = str(new_path.as_posix())
            elif "report_dir" == kwarg:
                continue
            elif name is not None:
                new_path = Path(path, name)
                full_path = str(Path(new_path).as_posix())
                new_files[kwarg] = full_path
                new_files[kwarg] = str(new_path.as_posix())
        return new_files

    def check_status(self) -> Dict[str, bool]:
        """Check the status of the files.
        :return: A dictionary of the files and whether or not they exist.
        """
        bools = {}
        files = self()
        for filename in files:
            file = files[filename]
            bools[filename] = Path(file).exists()
        return bools

    def __hash__(self):
        return int(my_hash(self), 16)
