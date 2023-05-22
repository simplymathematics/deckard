import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from copy import deepcopy
import numpy as np
import pandas as pd
import yaml

from ..utils import my_hash, to_dict

logger = logging.getLogger(__name__)

__all__ = ["FileConfig"]


@dataclass
class FileConfig:
    reports: str = "reports"
    data_dir: str = "data"
    model_dir: str = "models"
    attack_dir = "attacks"
    name: str = None
    stage: str = None
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
        # self._target_ = "deckard.base.files.FileConfig"
        needs = [reports, data_dir, model_dir, attack_dir]
        for need in needs:
            assert need is not None, f"Need to specify {need}"
        files.update(kwargs)
        self.reports = str(Path(reports).as_posix())
        self.data_dir = str(Path(data_dir).as_posix())
        self.model_dir = str(Path(model_dir).as_posix())
        self.attack_dir = str(Path(attack_dir).as_posix())
        self.data_type = data_type
        self.model_type = model_type
        self.attack_type = attack_type
        self.directory = directory
        self.name = name
        self.stage = stage
        self.files = self._set_filenames(**files)
        logger.debug(f"FileConfig init: {self.files}")

    def __call__(self):
        files = dict(self.get_filenames())
        return files

    def get_filenames(self):
        files = deepcopy(self.files)
        return files

    def _set_filenames(self, **kwargs):
        name = kwargs.pop("name", self.name)
        stage = kwargs.pop("stage", self.stage)
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
        if directory is not None:
            reports = str(Path(directory, reports).as_posix())
            data_dir = str(Path(directory, data_dir).as_posix())
            model_dir = str(Path(directory, model_dir).as_posix())
            attack_dir = str(Path(directory, attack_dir).as_posix())
        else:
            reports = str(Path(reports).as_posix())
            data_dir = str(Path(data_dir).as_posix())
            model_dir = str(Path(model_dir).as_posix())
            attack_dir = str(Path(attack_dir).as_posix())
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
            if "data_file" == kwarg:
                new_path = Path(data_dir, name)
                if new_path.suffix != data_type:
                    new_path = Path(data_dir, Path(name).stem + data_type)
                new_files[kwarg] = str(new_path.as_posix())
            elif "model_file" == kwarg:
                new_path = Path(model_dir, name)
                if new_path.suffix != model_type:
                    new_path = Path(model_dir, Path(name).stem + model_type)
                new_files[kwarg] = str(new_path.as_posix())
            elif "attack_file" == kwarg:
                new_path = Path(attack_dir, name)
                if new_path.suffix != attack_type:
                    new_path = Path(attack_dir, Path(name).stem + attack_type)
                new_files[kwarg] = str(new_path.as_posix())
            elif "directory" == kwarg:
                new_path = Path(name)
                new_files[kwarg] = str(new_path.as_posix())
            elif "name" == kwarg or "stage" == kwarg:
                continue
            elif "_type" in kwarg:
                continue
            elif "_dir" in kwarg:
                continue
            elif "reports" == kwarg:
                continue
            else:
                suffix = Path(name).suffix
                stem = Path(name).stem
                name = stem + suffix
                if str(name).endswith(suffix):
                    new_path = Path(path, name)
                else:
                    new_path = Path(path, name + suffix)
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

    def load(self) -> Dict[str, Any]:
        """Load the files from the file system.
        :return: A dictionary of the loaded files.
        """
        contents = {}
        files = self()
        for filename in files:
            file = files[filename]
            logger.info(f"Loading {filename} from {file}")
            if file is not None and Path(file).is_file():
                suffix = Path(file).suffix
                if suffix in ".json":
                    with open(file, "r") as f:
                        try:
                            new_path = json.load(f)
                        except json.decoder.JSONDecodeError:
                            logger.warning(
                                f"Could not load {filename} from {file}. Deleting file.",
                            )
                            Path(file).unlink()
                elif suffix in ".csv":
                    new_path = pd.read_csv(file)
                    new_path = new_path.to_numpy()
                elif suffix in ".npy":
                    new_path = np.load(file)
                elif suffix in ".pkl" or suffix in ".pickle":
                    with open(file, "rb") as f:
                        new_path = pickle.load(f)
                elif suffix in ".npz":
                    new_path = np.load(file)
                elif suffix in ".txt":
                    with open(file, "r") as f:
                        f.read()
                        new_path = f
                elif suffix in ".yaml":
                    with open(file, "r") as f:
                        new_path = yaml.load(f, Loader=yaml.FullLoader)
                elif suffix in ".db":
                    raise NotImplementedError(
                        f"File type not supported: {Path(file).suffix}. You can add support for this file type by adding a new if statement to the load method of the FileConfig class in {__file__}.",
                    )
                elif suffix in ".h5":
                    import tensorflow as tf

                    new_path = tf.keras.models.load_model(file)
                elif suffix in [".pt", ".pth"]:
                    import torch

                    new_path = torch.load(file)
                else:
                    raise NotImplementedError(
                        f"File type not supported: {Path(file).suffix}. You can add support for this file type by adding a new if statement to the load method of the FileConfig class in {__file__}.",
                    )
                if str(filename).endswith("_file"):
                    filename = filename.split("_file")[0]
                contents[filename] = new_path
            else:
                pass
        return contents

    def get_path(self):
        """Get the path to the file.
        :param stage: The stage of the experiment.
        :param name: The name of the experiment.
        :return: The path to the file.
        """
        files = self()
        return files["reports"]

    def save(self, **kwargs) -> List[str]:
        """
        Saves files to disk. Returns a list of the files saved.
        :param kwargs: The files to save. The key is the name of the file, and the value is the content to save.
        :return: A list of the files saved.
        """
        files = self()
        outputs = {}
        filenames = list(files.keys())
        for file in filenames:
            if file not in kwargs:
                files.pop(file)
        assert (
            kwargs.keys() == files.keys()
        ), f"Expected {files.keys()}, but got {kwargs.keys()}."
        for name in files:
            file = files[name]
            contents = kwargs[name]
            logger.info(f"Saving {name} to {file}.")
            if isinstance(contents, np.ndarray):
                logger.warning("Converting numpy array to list before saving to json.")
                contents = contents.tolist()
            elif isinstance(contents, pd.DataFrame):
                logger.warning(
                    "Converting pandas DataFrame to list before saving to json.",
                )
                contents = contents.to_dict(orient="records")
            elif isinstance(contents, pd.Series):
                logger.warning(
                    "Converting pandas Series to list before saving to json.",
                )
                contents = contents.to_dict(orient="records")
            elif isinstance(contents, (list, str, int, float, bool, type(None))):
                pass
            elif isinstance(contents, dict):
                contents = to_dict(contents)
            # elif isinstance(contents, (torch.Tensor, torch.nn.Module)):
            #     raise NotImplementedError(f"Saving {type(contents)} to json is not supported.")
            # elif isinstance(contents, (tf.keras.Model, tf.keras.layers.Layer)):
            #     raise NotImplementedError(f"Saving {type(contents)} to json is not supported.")
            # elif isinstance(contents, (tf.Tensor, tf.Variable)):
            #     raise NotImplementedError(f"Saving {type(contents)} to json is not supported.")
            else:
                raise ValueError(
                    f"Contents of type {type(contents)} cannot be saved to json.",
                )
            suffix = str(Path(file).suffix)
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            if suffix in ".json":
                contents = eval(str(contents))
                with open(file, "w") as f:
                    json.dump(contents, f)

            elif suffix in ".csv":
                pd.DataFrame(contents).to_csv(file, index=False)
            elif suffix in ".npy":
                np.save(file, contents)
            elif suffix in ".pkl" or ".pickle":
                with open(file, "wb") as f:
                    pickle.dump(contents, f)
            elif suffix in ".npz":
                np.savez(file, **contents)
            elif suffix in ".txt":
                with open(file, "w") as f:
                    f.write(contents)
            elif suffix in [".yml", ".yaml"]:
                contents = eval(json.dumps(contents))
                yaml.dump(contents, file)
            elif suffix in ".db":
                raise NotImplementedError(
                    f"File type not supported: {Path(file).suffix}. You can add support for this file type by adding a new if statement to the save method of the FileConfig class in {__file__}.",
                )
            elif file in ".h5":
                import tensorflow as tf

                tf.keras.models.save_model(contents, file)
            elif suffix in [".pt", ".pth"]:
                import torch

                torch.save(contents, file)
            else:
                raise NotImplementedError(
                    f"File type not supported: {Path(file).suffix}. You can add support for this file type by adding a new if statement to the save method of the FileConfig class in {__file__}.",
                )
            assert Path(file).exists(), f"File {file} was not saved."
            outputs[name] = file
        return outputs

    def __hash__(self):
        return int(my_hash(self), 16)
