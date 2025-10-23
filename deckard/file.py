from dataclasses import dataclass, field
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

default_placeholder_dict = {
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "experiment_name": "experiment_{timestamp}",
    "hash": None,  # Placeholder for hash; to be filled in as needed
}


@dataclass
class FileConfig(ConfigBase):
    """Configuration for file paths used in the experiment."""

    data_file: str = field(
        default_factory=str, metadata={"help": "Path to the data file."}
    )
    model_file: str = field(
        default_factory=str, metadata={"help": "Path to the model file."}
    )
    defense_file: str = field(
        default_factory=str, metadata={"help": "Path to the defense file."}
    )
    attack_file: str = field(
        default_factory=str, metadata={"help": "Path to the attack file."}
    )
    log_file: str = field(
        default_factory=str, metadata={"help": "Path to the log file."}
    )
    training_predictions_file: str = field(
        default_factory=str, metadata={"help": "Path to the training predictions file."}
    )
    test_predictions_file: str = field(
        default_factory=str, metadata={"help": "Path to the test predictions file."}
    )
    attack_predictions_file: str = field(
        default_factory=str, metadata={"help": "Path to the attack predictions file."}
    )
    score_file: str = field(
        default_factory=str, metadata={"help": "Path to the score file."}
    )
    experiment_name: str = field(
        default_factory=str, metadata={"help": "Name of the experiment."}
    )

    def __post_init__(self):
        # Assert that all files in all_files are attributes of the class
        for file_attr in all_files:
            assert hasattr(
                self,
                file_attr,
            ), f"FileConfig is missing attribute: {file_attr}"
        self.experiment_name = self._replace_placeholders(
            self.experiment_name,
            placeholder_dict=default_placeholder_dict,
        )
        self._resolve_paths()
        super().__post_init__()

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
        if self.experiment_name:
            path = str(path).replace("{experiment_name}", self.experiment_name)
        if path is None or len(path) == 0:
            return None
        if "{hash}" in path:
            if placeholder_dict.get("hash", None) is None:
                dummy_hash = hashlib.md5(self.experiment_name.encode()).hexdigest()
                placeholder_dict["hash"] = dummy_hash
            path = str(path).replace("{hash}", placeholder_dict["hash"])
        if "{experiment_name}" in path:
            path = str(path).replace("{experiment_name}", self.experiment_name)
        if "{timestamp}" in path:
            path = str(path).replace("{timestamp}", timestamp)
        for placeholder, value in placeholder_dict.items():
            path = str(path).replace("{" + placeholder + "}", str(value))
        return path

    def _resolve_paths(self, placeholder_dict={}) -> None:
        """Resolve file paths by replacing placeholders with actual values."""
        for file_attr in all_files:
            file_path = getattr(self, file_attr)
            if len(file_path) > 0:
                resolved_path = self._replace_placeholders(file_path, placeholder_dict)
                setattr(self, file_attr, resolved_path)

    def __call__(self) -> dict:
        """Return a dictionary of file paths."""
        file_dict = {}
        for file_attr in all_files:
            file_path = getattr(self, file_attr)
            if len(file_path) > 0:
                file_dict[file_attr] = file_path
        return file_dict

    # Define the len method to count non-None file attributes
    def __len__(self) -> int:
        count = 0
        for file_attr in all_files:
            if getattr(self, file_attr) is not None:
                count += 1
        return count
