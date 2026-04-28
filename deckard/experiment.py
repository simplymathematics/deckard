import logging
import warnings
import hashlib
from dataclasses import dataclass
from typing import List, Union, Literal
from omegaconf import DictConfig, OmegaConf
import os


import numpy as np
from pathlib import Path
from hydra.utils import instantiate

from .data import DataConfig, DataPipelineConfig
from .model import ModelConfig
from .model.defend import DefenseConfig
from .attack import AttackConfig
from .score import ScorerDictConfig
from .file import FileConfig, data_files, model_files, attack_files
from .utils import ConfigBase

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)




DECKARD_CONFIG_DIR = os.environ.get("DECKARD_CONFIG_DIR", "config")
DECKARD_DEFAULT_CONFIG_FILE = os.environ.get(
    "DECKARD_DEFAULT_CONFIG_FILE",
    "default_experiment.yaml",
)


# hydra_plugins/file_resolver.py
import os
from pathlib import Path
import yaml
from omegaconf import OmegaConf

# hydra_plugins/file_resolver.py
import os
from pathlib import Path
import yaml
from omegaconf import OmegaConf

def _load_yaml_file(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)

def _file_resolver(arg: str):
    """
    Usage:
      ${file:search/rf.yaml:model_search}
      ${file:./configs/search/rf.yaml:model_search.subkey}
      ${file:/abs/path/to/file.yaml}       -> returns whole file
    """
    if not arg:
        raise ValueError("file resolver requires an argument like 'path/to/file.yaml[:key]'")

    # split into path and optional key (only first ':' splits, keys may contain '.')
    if ":" in arg:
        path_part, key_part = arg.split(":", 1)
        key_part = key_part.strip()
    else:
        path_part, key_part = arg, None
    path = Path(DECKARD_CONFIG_DIR, path_part)
    if not path.exists():
        raise FileNotFoundError(f"file resolver: file not found: {path_part} in working dir {os.getcwd()}")
    
    data = _load_yaml_file(path)
    # if user requested a nested key, walk the dict using dot-splitting
    if key_part:
        parts = key_part.split(".")
        cur = data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                raise KeyError(f"file resolver: key '{key_part}' not found in {path}")
        data = cur
    data = OmegaConf.create(data)
    # Return as an OmegaConf node so structured content is preserved
    return data

# Register resolver with OmegaConf (Hydra will pick up this plugin module automatically)
OmegaConf.register_new_resolver("file", _file_resolver, replace=True, use_cache=True)


def _merge_resolver(*args):
    """
    Merge multiple OmegaConf or dict objects into a single OmegaConf dict.
    Usage:
      ${merge:${file:search/rf.yaml:model_search}, ${file:search/class_labels.yaml:model_search}}
    """
    merged = OmegaConf.create()
    for arg in args:
        # Resolve any interpolations
        obj = OmegaConf.to_container(OmegaConf.create(arg), resolve=True)
        merged = OmegaConf.merge(merged, obj)
    return OmegaConf.create(merged)

OmegaConf.register_new_resolver("merge", _merge_resolver, replace=True)


class ExperimentConfig(ConfigBase):
    data: Union[DataConfig, DataPipelineConfig]
    experiment_name: str = "{hash}"
    model: ModelConfig = None
    defense: DefenseConfig = None
    attack: AttackConfig = None
    files: FileConfig = None
    score: ScorerDictConfig = None
    random_state: int = 42
    library: Literal["sklearn", "tensorflow", "pytorch"] = "sklearn"
    classifier: Union[str, bool] = True

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
        if not hasattr(self, "_target_") or self._target_ is None:
            self._target_ = "deckard.experiment.ExperimentConfig"
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
            if hasattr(self.data, "_target_"):
                self.data = instantiate(self.data)
            elif isinstance(self.data, DictConfig):
                data_dict = OmegaConf.to_container(self.data)
            elif isinstance(self.data, str):
                data_dict = DataConfig.from_yaml(self.data).to_dict()
            elif isinstance(self.data, ConfigBase):
                data_dict = self.data.to_dict()
            elif isinstance(self.data, dict):
                data_dict = self.data
            else:
                raise ValueError(f"Unsupported type for data: {type(self.data)}")
            if "pipeline" not in data_dict:
                self.data = DataConfig(**data_dict)
            else:
                self.data = DataPipelineConfig(**data_dict)

        assert isinstance(
            self.data,
            DataConfig,
        ), f"data must be an instance of DataConfig. Got {type(self.data)}"
        self.data.__post_init__()
        if not hasattr(self, "classifier"):
            self.classifier = self.data.classifier
        else:
            assert (
                self.classifier == self.data.classifier
            ), f"classifier in experiment must match data.classifier. Got {self.classifier} vs {self.data.classifier}"
        if self.defense is not None:                    
           
            if isinstance(self.defense, DefenseConfig):
                pass
            else:
                if isinstance(self.defense, DictConfig):
                    defense_dict = OmegaConf.to_container(self.defense)
                elif isinstance(self.defense, str):
                    defense_dict = DefenseConfig.from_yaml(self.defense).to_dict()
                elif isinstance(self.defense, ConfigBase):
                    defense_dict = self.defense.to_dict()
                elif isinstance(self.defense, dict):
                    defense_dict = OmegaConf.to_container(OmegaConf.create(self.defense))
                else:
                    raise ValueError(
                        f"Unsupported type for defense: {type(self.defense)}",
                    )
                if "_target_" not in defense_dict:
                    self.defense = DefenseConfig(**defense_dict)
                else:
                    self.defense = instantiate(self.defense)
            assert isinstance(
                self.defense,
                DefenseConfig,
            ), "defense must be an instance of DefenseConfig"
            self.defense.__post_init__()
        if self.model is not None:
            if self.defense is not None:
                self.model.defense = self.defense
            if isinstance(self.model, ModelConfig):
                pass
            else:
                if hasattr(self.model, "_target_"):
                    self.model = instantiate(self.model)
                elif isinstance(self.model, DictConfig):
                    model_dict = OmegaConf.to_container(self.model)
                    self.model = ModelConfig(**model_dict)
                elif isinstance(self.model, str):
                    model_dict = ModelConfig.from_yaml(self.model).to_dict()
                    self.model = ModelConfig(**model_dict)
                elif isinstance(self.model, ConfigBase):
                    model_dict = self.model.to_dict()
                    self.model = ModelConfig(**model_dict)
                elif isinstance(self.model, dict):
                    model_dict = self.model
                    self.model = ModelConfig(**model_dict)
                else:
                    raise ValueError(f"Unsupported type for model: {type(self.model)}")
            assert isinstance(
                self.model,
                ModelConfig,
            ), "model must be an instance of ModelConfig"
            
            self.model.__post_init__()
            if self.classifier is None:
                self.classifier = self.model.classifier
            else:
                assert (
                    self.classifier == self.model.classifier
                ), f"classifier in experiment must match model.classifier. Got {self.classifier} vs {self.model.classifier}"
        if self.attack is not None:
            if isinstance(self.attack, AttackConfig):
                pass
            else:
                if isinstance(self.attack, DictConfig):
                    attack_dict = OmegaConf.to_container(self.attack)
                elif isinstance(self.attack, str):
                    attack_dict = AttackConfig.from_yaml(self.attack).to_dict()
                elif isinstance(self.attack, ConfigBase):
                    attack_dict = self.attack.to_dict()
                elif isinstance(self.attack, dict):
                    attack_dict = OmegaConf.to_container(OmegaConf.create(self.attack))
                else:
                    raise ValueError(
                        f"Unsupported type for attack: {type(self.attack)}",
                    )
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
        # Initialize FileConfig, ensuring experiment_name is set
        if self.files is None:
            self.files = FileConfig()
        elif isinstance(self.files, FileConfig):
            self.files.__post_init__()
        elif isinstance(self.files, ConfigBase):
            file_dict = self.files.to_dict()
            self.files = FileConfig(**file_dict)
        elif isinstance(self.files, DictConfig):
            file_dict = OmegaConf.to_container(self.files)
            self.files = FileConfig(**file_dict)
        elif isinstance(self.files, str):
            file_dict = FileConfig.from_yaml(self.files).to_dict()
            self.files = FileConfig(**file_dict)
        elif isinstance(self.files, dict):
            file_dict = self.files
            self.files = FileConfig(**file_dict)
        else:
            raise ValueError(f"Unsupported type for files: {type(self.files)}")
        assert isinstance(
            self.files,
            FileConfig,
        ), "file must be an instance of FileConfig"
        self.files.__post_init__()

        # Set scorers
        if isinstance(self.score, DictConfig):
            score_dict = OmegaConf.to_container(self.score)
            self.score = ScorerDictConfig(**score_dict)
        elif isinstance(self.score, ConfigBase):
            score_dict = self.score.to_dict()
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
        file_dict = self.files._get_file_dict()
        data_file_outputs = {
            file: getattr(self.files, file)
            for file in data_files
            if file in file_dict
        }
        model_file_outputs = {
            file: getattr(self.files, file)
            for file in model_files
            if file in file_dict
        }
        attack_file_outputs = {
            file: getattr(self.files, file)
            for file in attack_files
            if file in file_dict
        }
        if (
            "data_file" in data_file_outputs
            and Path(
                data_file_outputs["data_file"],
            ).exists()
        ):
            self.data = self.load_object(
                data_file_outputs["data_file"],
            )
        else:
            self.data(**data_file_outputs)
        assert hasattr(
            self.data,
            "X_train",
        ), f"data must return an object with X_train attribute"
        assert hasattr(
            self.data,
            "y_train",
        ), "data must return an object with y_train attribute"
        assert hasattr(
            self.data,
            "X_test",
        ), "data must return an object with X_test attribute"
        assert hasattr(
            self.data,
            "y_test",
        ), "data must return an object with y_test attribute"
        assert hasattr(
            self.data,
            "score_dict",
        ), "data must have score_dict attribute after loading"
        scores.update(**self.data.score_dict)
        if self.model:
            self.model(data=self.data, **model_file_outputs)
            assert hasattr(
                self.model,
                "training_predictions",
            ), "model must have training_predictions attribute after training"
            assert hasattr(
                self.model,
                "predictions",
            ), "model must have predictions attribute after training"
            assert hasattr(
                self.model,
                "score_dict",
            ), "model must have score_dict attribute after training"
            scores.update(**self.model.score_dict)
        else:
            logger.info("No model config provided, skipping model training.")
            self.model = None
        if self.attack:
            self.attack(
                data=self.data,
                model=self.model,
                **attack_file_outputs,
            )
            assert hasattr(
                self.attack,
                "attack",
            ), "attack must have attack attribute after training"
            assert hasattr(
                self.attack,
                "attack_predictions",
            ), "attack must have a predictions attribute after training"
            assert hasattr(
                self.attack,
                "score_dict",
            ), "attack must have score_dict attribute after training"
            scores.update(**self.attack.score_dict)
        else:
            logger.info("No attack config provided, skipping attack.")
        if self.score:
            custom_scores = self.score(
                data=self.data,
                model=self.model,
                attack=self.attack,
                mode="train",
                score_file=file_dict.get("score_file", None),
            )
            scores = {**scores, **custom_scores}
            # TODO: override existing score functions
        if "score_file" in file_dict and not Path(file_dict["score_file"]).exists():
            self.save_scores(scores, file_dict["score_file"])
        elif "score_file" in file_dict:
            old_scores = self.load_scores(file_dict["score_file"])
            new_scores = {**old_scores, **scores}
            self.save_scores(new_scores, file_dict["score_file"])
        else:
            logger.info("No score_file specified, skipping score saving.")
        return scores
    
