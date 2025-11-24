from dataclasses import dataclass, field
import inspect
import time
import hashlib
import logging
from pathlib import Path


from .utils import ConfigBase

logger = logging.getLogger(__name__)

data_files = ["data_file", "score_file"]
model_files = [
    "model_file",
    "training_predictions_file",
    "test_predictions_file",
    "score_file",
]
defense_files = [
    "training_predictions_file",
    "test_predictions_file",
    "score_file",
]
log_files = ["log_file"]
attack_files = [
    "attack_file",
    "attack_predictions_file",
    "score_file",
]
other_files = ["score_file", "params_file"]
all_files = (
    data_files + model_files + defense_files + log_files + attack_files + other_files
)

# make this an immutable default dict

default_placeholder_dict = {
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "experiment_name": "experiment_{timestamp}",
    "hash": None,  # Placeholder for hash; to be filled in as needed
}
immutable_placeholder_default = frozenset(default_placeholder_dict.items())


@dataclass
class FileConfig(ConfigBase):
    """Configuration for file paths used in the experiment."""
    data_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the data file."},
    )
    model_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the model file."},
    )
    defense_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the defense file."},
    )
    attack_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the attack file."},
    )
    log_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the log file."},
    )
    training_predictions_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the training predictions file."},
    )
    test_predictions_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the test predictions file."},
    )
    attack_predictions_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the attack predictions file."},
    )
    score_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the score file."},
    )
    params_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the params file."},
    )
    experiment_name: str = field(
        default_factory=str,
        metadata={"help": "Name of the experiment."},
    )
    log_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the log file."},
    )
    replace: dict = field(
        # default_factory=dict,
        metadata={"help": "Dictionary for placeholder replacements."},
        default=immutable_placeholder_default,
    )
    

    def __post_init__(self):
        # Assert that all files in all_files are attributes of the class
        # for file_attr in all_files:
        #     assert hasattr(
        #         self,
        #         file_attr,
        #     ), f"FileConfig is missing attribute: {file_attr}"
        super().__post_init__()
        self.experiment_name = self._replace_placeholders(
            path=self.experiment_name,
        )
        
        self._resolve_paths()
        

    def generate_file_hash(self, file_path: str) -> str:
        """
        Generate a hash for the object in the given file path.

        Args:
            file_path (str): The path to the file.

        Returns:
            int: The hash of the file contents.
        """
        # Using MD5 hash for simplicity; the impact of hash collisions is minimal here
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_hydra_job_num(self) -> int:
        """Get the Hydra job number from the environment variable."""
        import os

        job_num = os.getenv("HYDRA_JOB_NUM")
        if job_num is not None:
            return str(int(job_num))
        else:
            return "0"

    def _replace_placeholders(self, path: str) -> str:
        """Replace placeholders in the file path with actual values."""
        assert isinstance(path, str), f"Path must be a string. Got {type(path)}"
        placeholder_dict = dict(self.replace)
        assert isinstance(placeholder_dict, (dict, frozenset)), f"Placeholder dictionary must be a dict or frozenset. Got {type(placeholder_dict)}"
        # If frozenset, convert to dict
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if self.experiment_name:
            path = str(path).replace("{experiment_name}", self.experiment_name)
        if path is None or len(path) == 0:
            return None
        for placeholder, value in placeholder_dict.items():
            if placeholder == "{timestamp}":
                value = timestamp
            elif placeholder == "{hash}" and (value is None or len(value) == 0):
                if Path(path).exists():
                    value = self.generate_file_hash(path)
                else:
                  value = self.experiment_name
            elif placeholder == "{num}":
                value = self.get_hydra_job_num()
            elif placeholder == "experiment_name":
                value = self.experiment_name
            elif placeholder == "*":
                value = self.get_hydra_job_num()
            if "{" + placeholder + "}" in path:
                path = str(path).replace("{" + placeholder + "}", str(value))
            else:
                path = str(path).replace(placeholder, str(value))
        return path


    
    def _resolve_paths(self) -> None:
        """Resolve file paths by replacing placeholders with actual values."""
        for file_attr in all_files:
            file_path = getattr(self, file_attr)
            if file_path is not None and len(file_path) > 0:
                resolved_path = self._replace_placeholders(file_path)
                setattr(self, file_attr, resolved_path)
            else:
                logger.debug(f"File attribute {file_attr} is None or empty; skipping placeholder replacement.")

    def __call__(self) -> dict:
        """Return a dictionary of file paths."""
        file_dict = {}
        for file_attr in all_files:
            file_path = getattr(self, file_attr)
            if file_path is not None and len(file_path) > 0:
                file_path = self._replace_placeholders(file_path)
                file_dict[file_attr] = file_path
        return file_dict

    # Define the len method to count non-None file attributes
    def __len__(self) -> int:
        count = 0
        for file_attr in all_files:
            if getattr(self, file_attr) is not None:
                count += 1
        return count
