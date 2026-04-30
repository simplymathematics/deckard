from dataclasses import dataclass, field
import time
import hashlib
import logging
from typing import Dict, Optional


from .utils import ConfigBase

logger = logging.getLogger(__name__)

data_files = ["data_file", "score_file"]
model_files = [
    "model_file",
    "training_predictions_file",
    "test_predictions_file",
    "training_probabilities_file",
    "test_probabilities_file",
    "score_file",
]
defense_files = [
    "training_predictions_file",
    "test_predictions_file",
    "training_probabilities_file",
    "test_probabilities_file",
    "score_file",
]
log_files = ["log_file", "error_file"]
attack_files = [
    "attack_file",
    "attack_predictions_file",
    "score_file",
]
other_files = ["score_file", "params_file"]
all_files = (
    data_files + model_files + defense_files + log_files + attack_files + other_files
)


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
    error_file: str = field(
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
    training_probabilities_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the training probabilities file."},
    )
    test_probabilities_file: str = field(
        default_factory=str,
        metadata={"help": "Path to the test probabilities file."},
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
    replace: Dict[str, str] = field(
        metadata={"help": "Dictionary for placeholder replacements."},
        default_factory=dict,
    )

    def __post_init__(self):
        super().__post_init__()
        if self.replace is None:
            self.replace = {}
        elif not isinstance(self.replace, dict):
            self.replace = dict(self.replace)
        self._file_dict = self._get_file_dict()
        self._resolve_paths()

        for file in self._file_dict:
            setattr(self, file, self._file_dict[file])
        for k, v in self._file_dict.items():
            setattr(self, k, v)

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

    def get_hydra_job_num(self) -> str:
        """Get the Hydra job number from the environment variable."""
        import os

        job_num = os.getenv("HYDRA_JOB_NUM")
        if job_num is not None:
            return str(int(job_num))
        else:
            return "0"

    def _replace_placeholders(self, path: Optional[str]) -> Optional[str]:
        if path is None or len(path) == 0:
            return None

        assert isinstance(path, str)

        # Built-in placeholders (always applied)
        path = path.replace("{num}", self.get_hydra_job_num())
        path = path.replace("{timestamp}", time.strftime("%Y%m%d-%H%M%S"))
        path = path.replace("{hash}", str(hash(self)))
        path = path.replace("#", self.get_hydra_job_num())
        path = path.replace("*", self.get_hydra_job_num())

        # User-defined placeholders
        placeholder_dict = self.replace or {}
        if not isinstance(placeholder_dict, dict):
            placeholder_dict = dict(placeholder_dict)

        for placeholder, value in placeholder_dict.items():
            path = path.replace(placeholder, str(value))

        return path

    def _resolve_paths(self) -> None:
        """Resolve file paths by replacing placeholders with actual values."""
        for file_attr in all_files:
            file_path = getattr(self, file_attr)
            if file_path is not None and len(file_path) > 0:
                resolved_path = self._replace_placeholders(file_path)
                setattr(self, file_attr, resolved_path)
            else:
                logger.debug(
                    f"File attribute {file_attr} is None or empty; skipping placeholder replacement.",
                )

    def _get_file_dict(self) -> dict:
        """Return a dictionary of file paths."""
        file_dict = {}
        for file_attr in all_files:
            file_path = getattr(self, file_attr)
            if file_path is not None and len(file_path) > 0:
                file_path = self._replace_placeholders(file_path)
                file_dict[file_attr] = file_path
        return file_dict

    def __iter__(self):
        for path in self._file_dict:
            yield path

    # Define the len method to count non-None file attributes
    def __len__(self) -> int:
        count = 0
        for file_attr in all_files:
            if getattr(self, file_attr) is not None:
                count += 1
        return count


    def __hash__(self):
        return super().__hash__()
