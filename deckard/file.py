from pathlib import Path
from dataclasses import dataclass
import time
import hashlib
import logging
from .utils import ConfigBase

logger = logging.getLogger(__name__)


@dataclass
class FileConfig(ConfigBase):
    """
    Configuration class for managing experiment file and directory paths.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment, supports placeholders like "{hash}" and "{timestamp}".
    result_directory : str
        Directory to store experiment results.
    model_directory : str
        Directory to store model files.
    data_directory : str
        Directory to store data files.
    log_directory : str
        Directory to store log files.
    model_file : str
        Filename for the model file.
    data_file : str
        Filename for the data file.
    log_file : str
        Filename for the log file, supports placeholders.
    training_predictions_file : str
        Filename for training predictions.
    test_predictions_file : str
        Filename for test predictions.
    attack_file : str
        Filename for attack results.
    attack_training_predictions_file : str
        Filename for attack training predictions.
    attack_test_predictions_file : str
        Filename for attack test predictions.
    score_file : str
        Filename for scores.

    Methods
    -------
    __post_init__()
        Initializes directory paths as Path objects, creates directories if they do not exist,
        sets the experiment name using supported placeholders, and updates file attributes
        with resolved placeholders and directory paths.
    __call__(**kwargs)
        Allows updating attributes via keyword arguments, reinitializes configuration,
        and returns a dictionary of all file attributes with their resolved paths.

    Raises
    ------
    ValueError
        For invalid parameter values or missing directories.

    Examples
    --------
    config = FileConfig(experiment_name="{timestamp}")
    files = config()
    model_path = files["model_file"]
    log_path = files["log_file"]
    """

    experiment_name: str = "{hash}"
    result_directory: str = "results"
    model_directory: str = "models"
    data_directory: str = "data"
    log_directory: str = "logs"
    model_file: str = "model.pkl"
    data_file: str = "data.csv"
    log_file: str = "{experiment_name}.log"
    training_predictions_file: str = "train_predictions.pkl"
    test_predictions_file: str = "test_predictions.pkl"
    attack_file: str = "attack_results.pkl"
    attack_training_predictions_file: str = "attack_train_predictions.pkl"
    attack_test_predictions_file: str = "attack_test_predictions.pkl"
    score_file: str = "scores.json"

    def __hash__(self):
        """
        Computes a hash value for the instance based on non-private attributes.
        """
        return super().__hash__()

    def __post_init__(self):
        """
        Post-initialization method for setting up file and directory paths.

        Converts string directory paths to Path objects, creates directories if they do not exist,
        sets the experiment name using supported placeholders, and updates file attributes
        with resolved placeholders and directory paths.

        Side Effects
        ------------
        Ensures that result, model, data, and log directories exist.
        Updates file attributes with resolved paths.

        Raises
        ------
        ValueError
            If directory creation fails.
        """
        # Convert string paths to Path objects and create directories if they don't exist
        self.result_directory = Path(self.result_directory)
        self.model_directory = Path(self.model_directory)
        self.data_directory = Path(self.data_directory)
        self.log_directory = Path(self.log_directory)

        # Set experiment name
        if self.experiment_name is None or self.experiment_name == "":
            self.experiment_name = time.strftime("%Y%m%d-%H%M%S")
        elif self.experiment_name == "{timestamp}":
            self.experiment_name = time.strftime("%Y%m%d-%H%M%S")
        elif self.experiment_name == "{hash}":
            hash_source = str(
                {k: v for k, v in self.__dict__.items() if k != "experiment_name"}
            )
            self.experiment_name = hashlib.md5(hash_source.encode()).hexdigest()
        elif "{hash}" in self.experiment_name:
            hash_source = str(
                {k: v for k, v in self.__dict__.items() if k != "experiment_name"}
            )
            self.experiment_name = self.experiment_name.replace(
                "{hash}", hashlib.md5(hash_source.encode()).hexdigest()
            )
        elif "{timestamp}" in self.experiment_name:
            self.experiment_name = self.experiment_name.replace(
                "{timestamp}", time.strftime("%Y%m%d-%H%M%S")
            )
        # else: leave as is

        # Update _file attributes with placeholders and join with directories
        supported_placeholders = ["{experiment_name}", "{timestamp}", "{hash}"]
        used_directories = []
        for attr in self.__dataclass_fields__:
            if attr.endswith("_file"):
                current_value = getattr(self, attr)
                # Replace placeholders
                if any(ph in current_value for ph in supported_placeholders):
                    current_value = current_value.replace(
                        "{experiment_name}", self.experiment_name
                    )
                    current_value = current_value.replace(
                        "{timestamp}", time.strftime("%Y%m%d-%H%M%S")
                    )
                    current_value = current_value.replace(
                        "{hash}", hashlib.md5(current_value.encode()).hexdigest()
                    )
                # Join with directory
                directory_attr = attr.replace("_file", "_directory")
                if hasattr(self, directory_attr):
                    used_directories.append(directory_attr)
                    directory_value = getattr(self, directory_attr)
                    if isinstance(directory_value, Path):
                        setattr(self, attr, str(directory_value / current_value))
                    else:
                        setattr(self, attr, str(Path(directory_value) / current_value))
                else:
                    setattr(self, attr, current_value)
        # Remove unused directory attributes
        for attr in self.__dataclass_fields__:
            if attr.endswith("_directory") and attr not in used_directories:
                delattr(self, attr)
        # Ensure directories exist
        for directory in [
            self.result_directory,
            self.model_directory,
            self.data_directory,
            self.log_directory,
        ]:
            directory = Path(directory)
            if not directory.exists():
                logger.info(f"Creating directory: {directory}")
                directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, **kwargs):
        """
        Updates configuration attributes and returns resolved file paths.

        Parameters
        ----------
        **kwargs
            Keyword arguments to update configuration attributes.

        Returns
        -------
        dict
            Dictionary of all file attributes with their resolved paths.

        Side Effects
        ------------
        Ensures that the folders for each file exist.

        Raises
        ------
        ValueError
            If directory creation fails.
        """
        # Update attributes with new values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.__post_init__()
        _files = {
            attr: getattr(self, attr) for attr in dir(self) if attr.endswith("_file")
        }
        # Ensure that the folders for each _file exists:
        for attr, filepath in _files.items():
            file_path = Path(filepath)
            if not file_path.parent.exists():
                logger.info(f"Creating directory for {attr}: {file_path.parent}")
                file_path.parent.mkdir(parents=True, exist_ok=True)
        return _files
