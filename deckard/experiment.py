import logging
import warnings
import hashlib
from dataclasses import dataclass
from typing import List, Union, Literal
from omegaconf import DictConfig


import numpy as np
from pathlib import Path

from .data import DataConfig
from .model import ModelConfig
from .model.defend import DefenseConfig
from .attack import AttackConfig
from .score import ScorerDictConfig
from .file import FileConfig, data_files, model_files, defense_files, attack_files
from .utils import ConfigBase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class PlotConfig(ConfigBase):
    pass


class ExperimentConfig(ConfigBase):
    data: DataConfig
    experiment_name: str = "{hash}"
    model: ModelConfig = None
    defense: DefenseConfig = None
    attack: AttackConfig = None
    files: FileConfig = None
    score: ScorerDictConfig = None
    random_state: int = 42
    library: Literal["sklearn", "tensorflow", "pytorch"] = "sklearn"

    def set_device(self, device: Union[str, int] = "cpu"):
        """
        Set the computation device for the experiment based on the selected library.
        For TensorFlow, configures GPU/CPU usage.
        Args:
            device (Union[str, int]): Device to use ("cpu", "gpu", or GPU index).
        """
        if self.library == "tensorflow":
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if device == "cpu":
                tf.config.set_visible_devices([], "GPU")
                logger.info("Using CPU for TensorFlow")
            elif isinstance(device, str) and "gpu" in device.lower():
                if gpus:
                    try:
                        tf.config.set_visible_devices(gpus[0], "GPU")
                        tf.config.experimental.set_memory_growth(gpus[0], True)
                        logger.info(f"Using GPU for TensorFlow: {gpus[0]}")
                    except RuntimeError as e:
                        logger.error(e)
                else:
                    logger.warning("No GPU found, using CPU for TensorFlow")
            elif isinstance(device, int) and gpus and device < len(gpus):
                try:
                    tf.config.set_visible_devices(gpus[device], "GPU")
                    tf.config.experimental.set_memory_growth(gpus[device], True)
                    logger.info(f"Using GPU for TensorFlow: {gpus[device]}")
                except RuntimeError as e:
                    logger.error(e)
            else:
                logger.warning(
                    "Invalid device specified for TensorFlow, using default device.",
                )
        elif self.library == "pytorch":
            import torch

            if device == "cpu":
                torch_device = torch.device("cpu")
                logger.info("Using CPU for PyTorch")
            elif (
                isinstance(device, str)
                and "gpu" in device.lower()
                and torch.cuda.is_available()
            ):
                torch_device = torch.device("cuda:0")
                logger.info("Using GPU for PyTorch: cuda:0")
            elif (
                isinstance(device, int)
                and torch.cuda.is_available()
                and device < torch.cuda.device_count()
            ):
                torch_device = torch.device(f"cuda:{device}")
                logger.info(f"Using GPU for PyTorch: cuda:{device}")
            else:
                torch_device = torch.device("cpu")
                logger.warning("Invalid device specified for PyTorch, using CPU.")
            self.torch_device = torch_device
        else:
            logger.info("Device selection not supported for library: %s", self.library)

    def __post_init__(self):
        # Set random seed
        self.set_random_seed()
        # Set device
        if self.library not in ["sklearn"]:
            self.set_device()
        # Validate and initialize configs
        if self.data is None:
            raise ValueError("data must be provided")
        if isinstance(self.data, DictConfig):
            self.data = DataConfig(**self.data)
        elif isinstance(self.data, str):
            self.data = DataConfig.from_yaml(self.data)
        assert isinstance(
            self.data,
            DataConfig,
        ), f"data must be an instance of DataConfig. Got {type(self.data)}"
        self.data.__post_init__()
        if self.model is not None:
            if isinstance(self.model, DictConfig):
                self.model = ModelConfig(**self.model)
            assert isinstance(
                self.model,
                ModelConfig,
            ), "model must be an instance of ModelConfig"
            self.model.__post_init__()
        if self.defense is not None:
            if isinstance(self.defense, DictConfig):
                self.defense = DefenseConfig(**self.defense)
            assert isinstance(
                self.defense,
                DefenseConfig,
            ), "defense must be an instance of DefenseConfig"
            self.defense.__post_init__()
        if self.attack is not None:
            if isinstance(self.attack, DictConfig):
                self.attack = AttackConfig(**self.attack)
            assert isinstance(
                self.attack,
                AttackConfig,
            ), "attack must be an instance of AttackConfig"
            self.attack.__post_init__()
        # Set experiment name if not provided
        if self.experiment_name in [None, "", "{hash}", "*"]:
            config_list = [self.data]
            if self.model:
                config_list.append(self.model)
            if self.defense:
                config_list.append(self.defense)
            if self.attack:
                config_list.append(self.attack)
            self.experiment_name = self._hash_from_list(config_list)
            logger.info(f"Generated experiment name: {self.experiment_name}")
        else:
            logger.info(f"Using provided experiment name: {self.experiment_name}")

        if self.files is None:
            self.files = FileConfig(experiment_name=self.experiment_name)
        elif isinstance(self.files, FileConfig):
            self.files.experiment_name = self.experiment_name
            self.files.__post_init__()
        else:
            self.files = FileConfig(**self.files, experiment_name=self.experiment_name)
            assert isinstance(
                self.files,
                FileConfig,
            ), "file must be an instance of FileConfig"
            self.files.__post_init__()

    def set_random_seed(self):
        if self.library in ["sklearn"]:
            np.random.seed(self.random_state)
        elif self.library in ["tensorflow"]:
            import tensorflow as tf

            tf.random.set_seed(self.random_state)
        elif self.library in ["pytorch"]:
            import torch

            torch.manual_seed(self.random_state)
        else:
            raise ValueError(f"Unsupported library: {self.library}")
        # Set

    def _hash_from_list(self, config_list: List[ConfigBase]) -> str:
        """
        Generate a hash string from a list of ConfigBase objects.
        The hash is generated by concatenating the string representations of the configurations
        and computing the MD5 hash of the resulting string.
        Args:
            config_list (List[ConfigBase]): List of ConfigBase objects to generate the hash from.
        Returns:
            str: The generated hash string.
        """
        for conf in config_list:
            assert isinstance(
                conf,
                ConfigBase,
            ), "All items in config_list must be instances of ConfigBase"
            to_string = "".join(
                [
                    str(getattr(conf, attr))
                    for attr in dir(conf)
                    if not attr.startswith("_") and not callable(getattr(conf, attr))
                ],
            )
        return hashlib.md5(to_string.encode()).hexdigest()

    def __call__(
        self,
    ):
        # Initialize Scores
        scores = {}
        # Set random seed
        self.set_random_seed()
        # Set device
        if self.library not in ["sklearn"]:
            self.set_device()
        # Get file paths
        file_dict = self.files()
        data_file_outputs = {
            file: self.files._replace_placeholders(file_dict[file])
            for file in data_files
            if file in file_dict
        }
        model_file_outputs = {
            file: self.files._replace_placeholders(file_dict[file])
            for file in model_files
            if file in file_dict
        }
        defense_file_outputs = {
            file: file_dict[file] for file in defense_files if file in file_dict
        }
        attack_file_outputs = {
            file: file_dict[file] for file in attack_files if file in file_dict
        }

        self.data(**data_file_outputs)
        data = self.data
        assert hasattr(
            data,
            "X_train",
        ), "data must return an object with X_train attribute"
        assert hasattr(
            data,
            "y_train",
        ), "data must return an object with y_train attribute"
        assert hasattr(
            data,
            "X_test",
        ), "data must return an object with X_test attribute"
        assert hasattr(
            data,
            "y_test",
        ), "data must return an object with y_test attribute"
        scores.update(**data.score_dict)
        if self.model:
            self.model(data=data, **model_file_outputs)
            model = self.model
            assert hasattr(
                model,
                "training_predictions",
            ), "model must have training_predictions attribute after training"
            assert hasattr(
                model,
                "predictions",
            ), "model must have predictions attribute after training"
            assert hasattr(
                model,
                "score_dict",
            ), "model must have score_dict attribute after training"
            scores.update(**model.score_dict)
        elif self.defense:
            self.defense(data=data, **defense_file_outputs)
            model = self.defense
            assert hasattr(
                model,
                "training_predictions",
            ), "model must have training_predictions attribute after training"
            assert hasattr(
                model,
                "predictions",
            ), "model must have predictions attribute after training"
            assert hasattr(
                model,
                "score_dict",
            ), "model must have score_dict attribute after training"
            scores.update(**model.score_dict)
        else:
            logger.info("No model or defense config provided, skipping model training.")
            model = None
        if self.attack:
            self.attack(
                data=data,
                model=model,
                **attack_file_outputs,
            )
            attack = self.attack
            assert hasattr(
                attack,
                "attack",
            ), "attack must have attack attribute after training"
            assert hasattr(
                attack,
                "predictions",
            ), "attack must have a predictions attribute after training"
            assert hasattr(
                attack,
                "score_dict",
            ), "attack must have score_dict attribute after training"
            scores.update(**attack.score_dict)
        else:
            logger.info("No attack config provided, skipping attack.")
            attack = None
        # Assert that all files in file_dict exist

        for attr, filepath in file_dict.items():
            if filepath is None:
                continue
            resolved_path = self.files._replace_placeholders(filepath)
            assert Path(
                resolved_path,
            ).exists(), f"File {resolved_path} for {attr} does not exist."
        return scores
