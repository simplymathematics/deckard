import time
import logging
import warnings
from dataclasses import dataclass
import hashlib
from typing import List, Union, Literal
import argparse


import numpy as np
from pathlib import Path

from .data import DataConfig
from .model import ModelConfig
from .model.defend import DefenseConfig
from .attack import AttackConfig
from .score import ScorerDictConfig
from .file import FileConfig
from .utils import ConfigBase, create_parser_from_function

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

data_call_parser = create_parser_from_function(DataConfig.__call__, exclude=["self"])
model_call_parser = create_parser_from_function(
    ModelConfig.__call__, exclude=["self", "data"]
)
defense_call_parser = create_parser_from_function(
    DefenseConfig.__call__, exclude=["self", "data", "estimator"]
)
attack_call_parser = create_parser_from_function(
    AttackConfig.__call__, exclude=["self", "data", "estimator"]
)


data_call_kwargs = list(vars(data_call_parser.parse_known_args([])[0]).keys())
model_call_kwargs = list(vars(model_call_parser.parse_known_args([])[0]).keys())
defense_call_kwargs = list(vars(defense_call_parser.parse_known_args([])[0]).keys())
attack_call_kwargs = list(vars(attack_call_parser.parse_known_args([])[0]).keys())


@dataclass
class PlotConfig(ConfigBase):
    pass


class ExperimentConfig(ConfigBase):
    data_config: DataConfig
    experiment_name: str = "default_experiment"
    model_config: ModelConfig = None
    defense_config: DefenseConfig = None
    attack_config: AttackConfig = None
    file_config: FileConfig = None
    score_config: ScorerDictConfig = None
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
                    "Invalid device specified for TensorFlow, using default device."
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
        self.set_device()
        # Validate and initialize configs
        if self.data_config is None:
            raise ValueError("data_config must be provided")
        assert isinstance(
            self.data_config, DataConfig
        ), "data_config must be an instance of DataConfig"
        self.data_config.__post_init__()
        if self.model_config:
            assert isinstance(
                self.model_config, ModelConfig
            ), "model_config must be an instance of ModelConfig"
            self.model_config.__post_init__()
        if self.attack_config:
            assert isinstance(
                self.attack_config, AttackConfig
            ), "attack_config must be an instance of AttackConfig"
            self.attack_config.__post_init__()
        # Set experiment name if not provided
        if self.experiment_name in [None, "", "{hash}", "*"]:
            config_list = [self.data_config]
            if self.model_config:
                config_list.append(self.model_config)
            if self.defense_config:
                config_list.append(self.defense_config)
            if self.attack_config:
                config_list.append(self.attack_config)
            self.experiment_name = self._hash_from_config_list(config_list)
        if self.file_config is None:
            self.file_config = FileConfig(experiment_name=self.experiment_name)
        else:
            assert isinstance(
                self.file_config, FileConfig
            ), "file_config must be an instance of FileConfig"
            self.file_config.__post_init__()

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

    def _hash_from_config_list(self, config_list: List[ConfigBase]) -> str:
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
                conf, ConfigBase
            ), "All items in config_list must be instances of ConfigBase"
            to_string = "".join(
                [
                    str(getattr(conf, attr))
                    for attr in dir(conf)
                    if not attr.startswith("_") and not callable(getattr(conf, attr))
                ]
            )
        return hashlib.md5(to_string.encode()).hexdigest()

    def __call__(self, **kwargs):
        # Update any kwargs in file_config
        scores = {}
        if self.file_config is None:
            self.file_config = FileConfig(experiment_name=self.experiment_name)
        else:
            assert isinstance(
                self.file_config, FileConfig
            ), "file_config must be an instance of FileConfig"
            self.file_config.__post_init__()
        file_dict = self.file_config(**kwargs)

        # Set random seed
        self.set_random_seed()
        # Set device
        self.set_device()

        # Get call params from data parser
        data_call_params = vars(data_call_parser.parse_known_args([])[0])
        data = self.data_config(**data_call_params, **file_dict)
        assert hasattr(
            data, "X_train"
        ), "data_config must return an object with X_train attribute"
        assert hasattr(
            data, "y_train"
        ), "data_config must return an object with y_train attribute"
        assert hasattr(
            data, "X_test"
        ), "data_config must return an object with X_test attribute"
        assert hasattr(
            data, "y_test"
        ), "data_config must return an object with y_test attribute"
        scores.update(**data.score_dict)
        if self.model_config:
            model_call_params = vars(model_call_parser.parse_known_args([])[0])
            data, model = self.model_config(data=data, **model_call_params, **file_dict)
            assert hasattr(
                model, "training_predictions"
            ), "model must have training_predictions attribute after training"
            assert hasattr(
                model, "predictions"
            ), "model must have predictions attribute after training"
            assert hasattr(
                model, "score_dict"
            ), "model must have score_dict attribute after training"
            for k, v in data.score_dict.items():
                assert (
                    k in model.score_dict
                ), f"score {k} from data not found in model score_dict"
                assert (
                    v == model.score_dict[k]
                ), f"score {k} from data does not match model score_dict"
            scores.update(**model.score_dict)
        elif self.defense_config:
            defense_call_params = vars(defense_call_parser.parse_known_args([])[0])
            data, model = self.defense_config(**defense_call_params, **file_dict)
            assert hasattr(
                model, "training_predictions"
            ), "model must have training_predictions attribute after training"
            assert hasattr(
                model, "predictions"
            ), "model must have predictions attribute after training"
            assert hasattr(
                model, "score_dict"
            ), "model must have score_dict attribute after training"
            for k, v in data.score_dict.items():
                assert (
                    k in model.score_dict
                ), f"score {k} from data not found in model score_dict"
                assert (
                    v == model.score_dict[k]
                ), f"score {k} from data does not match model score_dict"
            scores.update(**model.score_dict)
        else:
            logger.info("No model or defense config provided, skipping model training.")
            model = None
        if self.attack_config:
            attack_call_params = vars(attack_call_parser.parse_known_args([])[0])
            data, model, attack = self.attack_config(
                data=data, model=model, **attack_call_params, **file_dict
            )
            assert hasattr(
                attack, "attack"
            ), "attack must have attack attribute after training"
            assert hasattr(
                attack, "attack_training_predictions"
            ), "attack must have attack_training_predictions attribute after training"
            assert hasattr(
                attack, "attack_predictions"
            ), "attack must have attack_predictions attribute after training"
            assert hasattr(
                attack, "attack_score_dict"
            ), "attack must have attack_score_dict attribute after training"
            for k, v in model.score_dict.items():
                assert (
                    k in attack.attack_score_dict
                ), f"score {k} from model not found in attack score_dict"
                assert (
                    v == attack.attack_score_dict[k]
                ), f"score {k} from model does not match attack score_dict"
            scores.update(**attack.attack_score_dict)
        else:
            logger.info("No attack config provided, skipping attack.")
            attack = None
        # Assert that all files in file_dict exist
        for attr, filepath in file_dict.items():
            file_path = Path(filepath)
            if not file_path.exists():
                logger.error(f"File {attr} does not exist: {file_path}")
                raise FileNotFoundError(f"File {attr} does not exist: {file_path}")
        return scores
