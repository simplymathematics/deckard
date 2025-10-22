from dataclasses import dataclass
import time
import hashlib
import logging
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
    "defense_file",
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
other_files = ["score_file"]
all_files = (
    data_files + model_files + defense_files + log_files + attack_files + other_files
)


@dataclass
class FileConfig(ConfigBase):
    """Configuration for file paths used in the experiment."""

    data_file: str = None
    model_file: str = None
    defense_file: str = None
    attack_file: str = None
    log_file: str = None
    training_predictions_file: str = None
    test_predictions_file: str = None
    attack_predictions_file: str = None
    score_file: str = None
    experiment_name: str = None

    def __post_init__(self):
        # Assert that all files in all_files are attributes of the class
        for file_attr in all_files:
            assert hasattr(
                self, file_attr
            ), f"FileConfig is missing attribute: {file_attr}"
        return super().__post_init__()

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

    def _replace_placeholders(self, path: str, placeholder_dict={}) -> str:
        """Replace placeholders in the file path with actual values."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = path.replace("{timestamp}", timestamp)
        if self.experiment_name:
            path = path.replace("{experiment_name}", self.experiment_name)
        for placeholder, value in placeholder_dict.items():
            path = path.replace("{" + placeholder + "}", str(value))
        return path

    def resolve_paths(self, placeholder_dict={}) -> None:
        """Resolve file paths by replacing placeholders with actual values."""
        for file_attr in all_files:
            file_path = getattr(self, file_attr)
            if file_path is not None:
                resolved_path = self._replace_placeholders(file_path, placeholder_dict)
                setattr(self, file_attr, resolved_path)

    def __call__(self) -> dict:
        """Return a dictionary of file paths."""
        file_dict = {}
        for file_attr in all_files:
            file_path = getattr(self, file_attr)
            if file_path is not None:
                file_dict[file_attr] = file_path
        return file_dict
