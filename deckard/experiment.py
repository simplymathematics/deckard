import logging
import warnings
import hashlib
from dataclasses import dataclass
from typing import List, Union, Literal
from omegaconf import DictConfig, OmegaConf


import numpy as np
from pathlib import Path
from hydra.utils import instantiate

from .data import DataConfig
from .model import ModelConfig
from .attack import AttackConfig
from .score import ScorerDictConfig
from .file import FileConfig, data_files, model_files, attack_files
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
    attack: AttackConfig = None
    files: FileConfig = None
    score: ScorerDictConfig = None
    random_state: int = 42
    classifier : str = None
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
        if isinstance(self.data, DataConfig):
            pass
        else:
            if isinstance(self.data, DictConfig):
                data_dict = OmegaConf.to_container(self.data)
            elif isinstance(self.data, str):
                data_dict = DataConfig.from_yaml(self.data).to_dict()
            elif isinstance(self.data, dict):
                data_dict = OmegaConf.to_container(OmegaConf.create(self.data))
            else:
                raise ValueError(f"Unsupported type for data: {type(self.data)}")
            
            if "_target_" not in data_dict:
                self.data = DataConfig(**data_dict)
            else:
                self.data = instantiate(self.data)
        assert isinstance(
            self.data,
            DataConfig,
        ), f"data must be an instance of DataConfig. Got {type(self.data)}"
        self.data.__post_init__()
        if self.classifier is None:
            self.classifier = self.data.classifier
        else:
            assert self.classifier == self.data.classifier, (
                f"classifier in experiment must match data.classifier. Got {self.classifier} vs {self.data.classifier}"
            )
        if self.model is not None:
            if isinstance(self.model, ModelConfig):
                pass
            else:
                if isinstance(self.model, DictConfig):
                    model_dict = OmegaConf.to_container(self.model)
                elif isinstance(self.model, str):
                    model_dict = ModelConfig.from_yaml(self.model).to_dict()
                elif isinstance(self.model, dict):
                    model_dict = OmegaConf.to_container(OmegaConf.create(self.model))
                else:
                    raise ValueError(f"Unsupported type for model: {type(self.model)}")
                if "_target_" not in model_dict:
                    self.model = ModelConfig(**model_dict)
                else:
                    self.model = instantiate(self.model)
            assert isinstance(
                self.model,
                ModelConfig,
            ), "model must be an instance of ModelConfig"
            self.model.__post_init__()
            if self.classifier is None:
                self.classifier = self.model.classifier
            else:
                assert self.classifier == self.model.classifier, (
                    f"classifier in experiment must match model.classifier. Got {self.classifier} vs {self.model.classifier}"
                )
        if self.attack is not None:
            if isinstance(self.attack, AttackConfig):
                pass
            else:
                if isinstance(self.attack, DictConfig):
                    attack_dict = OmegaConf.to_container(self.attack)
                elif isinstance(self.attack, str):
                    attack_dict = AttackConfig.from_yaml(self.attack).to_dict()
                elif isinstance(self.attack, dict):
                    attack_dict = OmegaConf.to_container(OmegaConf.create(self.attack))
                else:
                    raise ValueError(f"Unsupported type for attack: {type(self.attack)}")
                if "_target_" not in attack_dict:
                    self.attack = AttackConfig(**attack_dict)
                else:
                    self.attack = instantiate(self.attack)
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
            if self.attack:
                config_list.append(self.attack)
            if self.score:
                config_list.append(self.score)
            self.experiment_name = self._hash_from_list(config_list)
            logger.info(f"Generated experiment name: {self.experiment_name}")
        else:
            logger.info(f"Using provided experiment name: {self.experiment_name}")
        # Initialize FileConfig, ensuring experiment_name is set
        if self.files is None:
            self.files = FileConfig(experiment_name=self.experiment_name)
        elif isinstance(self.files, FileConfig):
            self.files.experiment_name = self.experiment_name
            self.files.__post_init__()
        elif isinstance(self.files, DictConfig):
            file_dict = OmegaConf.to_container(self.files)
            file_dict["experiment_name"] = self.experiment_name
            self.files = FileConfig(**file_dict)
        elif isinstance(self.files, str):
            file_dict = FileConfig.from_yaml(self.files).to_dict()
            file_dict["experiment_name"] = self.experiment_name
            self.files = FileConfig(**file_dict)
        elif isinstance(self.files, dict):
            file_dict = self.files
            file_dict["experiment_name"] = self.experiment_name
            self.files = FileConfig(**file_dict)
        else:
            raise ValueError(f"Unsupported type for files: {type(self.files)}")
        assert isinstance(
            self.files,
            FileConfig,
        ), "file must be an instance of FileConfig"
        self.files.__post_init__()
        assert self.files.experiment_name == self.experiment_name, (
            f"files.experiment_name must match experiment_name. Got {self.files.experiment_name} vs {self.experiment_name}",
        )
    
        # Set scorers
        if isinstance(self.score, DictConfig):
            score_dict = OmegaConf.to_container(self.score)
            self.score = ScorerDictConfig(**score_dict)
        elif isinstance(self.score, str):
            score_dict = ScorerDictConfig.from_yaml(self.score).to_dict()
            self.score = ScorerDictConfig(**score_dict)
        elif isinstance(self.score, dict):
            score_dict = self.score
            self.score = ScorerDictConfig(**score_dict)
        elif self.score is None:
            pass
        else:
            raise ValueError(f"Unsupported type for score: {type(self.score)}")
        
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
        else:
            logger.info("No model config provided, skipping model training.")
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
            else:
                filepath = self.files._replace_placeholders(filepath)
                assert Path(filepath).exists(), f"File {filepath} for {attr} does not exist."
            # 
        if self.score:
            custom_scores = self.score(
                data=data,
                model=model,
                attack=attack,
                mode="train",
                score_file=file_dict.get("score_file", None)
            )
            scores = {**scores, **custom_scores}
            # TODO: override existing score functions
        return scores
