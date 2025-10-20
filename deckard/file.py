from pathlib import Path
from dataclasses import dataclass
import time
import hashlib
import logging
from .utils import ConfigBase

logger = logging.getLogger(__name__)

data_files = ["data_file", "data_score_file",]
model_files =["model_file", "score_file", "training_predictions_file", "test_predictions_file", "score_file"]
defense_files = ["defense_file", "defense_score_file", "training_predictions_file", "test_predictions_file", "defense_score_file"]
log_files = ["log_file",]
attack_files = ["attack_file", "attack_training_predictions_file", "attack_test_predictions_file", "attack_score_file"]
other_files = ["score_file",]
all_files = data_files + model_files + defense_files + log_files + attack_files + other_files
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
    result_directory: str = None
    model_directory: str = None
    data_directory: str = None
    attack_directory: str = None
    log_directory: str = None
    model_file: str = None 
    data_file: str = None 
    log_file: str = None
    training_predictions_file: str = None
    test_predictions_file: str = None
    attack_file: str = None
    attack_training_predictions_file: str = None
    attack_test_predictions_file: str = None
    score_file: str = None

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
        self.result_directory = Path(self.result_directory) if self.result_directory else None
        self.model_directory = Path(self.model_directory) if self.model_directory else None
        self.data_directory = Path(self.data_directory) if self.data_directory else None
        self.log_directory = Path(self.log_directory) if self.log_directory else None

        # Set experiment name
        self.experiment_name = self._replace_placeholders(self.experiment_name)
        self._resolve_placeholders_in_files()
        self._drop_unused_directories()
        
    def _drop_unused_directories(self):
        # Remove directory attributes if no corresponding files are defined
        if not any(getattr(self, f, None) for f in model_files):
            self.model_directory = None
        if not any(getattr(self, f, None) for f in data_files):
            self.data_directory = None
        if not any(getattr(self, f, None) for f in attack_files):
            self.attack_directory = None
        if not any(getattr(self, f, None) for f in log_files):
            self.log_directory = None
        if not any(getattr(self, f, None) for f in attack_files):
            self.attack_directory = None

    def _resolve_placeholders_in_files(self):
        # Update file attributes with resolved placeholders and directory paths
        for attr  in self.__dataclass_fields__.keys():
            if str(attr).endswith("_file"):
                file_value = getattr(self, attr)
                if file_value:
                    resolved_file = self._replace_placeholders(file_value)
                    # Prepend directory if applicable
                    if attr in model_files and self.model_directory:
                        resolved_file = str(self.model_directory / resolved_file)
                    elif attr in data_files and self.data_directory:
                        resolved_file = str(self.data_directory / resolved_file)
                    elif attr in log_files and self.log_directory:
                        resolved_file = str(self.log_directory / resolved_file)
                    elif attr in attack_files and self.attack_directory:
                        resolved_file = str(self.attack_directory / resolved_file)
                    setattr(self, attr, resolved_file)

    def _replace_placeholders(self, file):
        """
        Generates the experiment name by replacing supported placeholders.

        Supported placeholders:
        - "{hash}": A unique hash based on the instance's attributes.
        - "{timestamp}": Current timestamp in seconds.
        - "{seed}": Random seed if applicable.

        Side Effects
        ------------
        Updates the experiment_name attribute with the resolved name.
        """
        if "{hash}" in file:
            hash_value = hashlib.md5(str(self.__hash__()).encode()).hexdigest()
            file = file.replace("{hash}", hash_value)
        if "{experiment_name}" in file:
            file = file.replace("{experiment_name}", self.experiment_name)
        if "{timestamp}" in file:
            timestamp = str(int(time.time()))
            file = file.replace("{timestamp}", timestamp)
        if "{seed}" in file:
            file = file.replace("{seed}", str(self.random_state))
        return file
    
    def _prepend_directory(self, file_attr: str, directory_attr: str) -> str:
        """
        Prepends the appropriate directory path to the given file attribute.

        Parameters
        ----------
        file_attr : str
            The name of the file attribute.
        directory_attr : str
            The name of the directory attribute.

        Returns
        -------
        str
            The full path with the directory prepended, or the original file name if no directory is set.
        """
        file_value = getattr(self, file_attr)
        directory_value = getattr(self, directory_attr)
        if file_value and directory_value:
            return str(Path(directory_value) / file_value)
        return file_value
    
    def __call__(self):
        """
        Allows updating attributes via keyword arguments, reinitializes configuration,
        and returns a dictionary of all file attributes with their resolved paths.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to update attributes of the instance.
        Returns
        -------
        dict
            A dictionary containing all file attributes with their resolved paths.
        Raises
        ------
        ValueError
            If any of the file attributes are not properly initialized.
        """
        
        # Prepend relevant directories to file attributes
        files = {}
        for attr in self.__dataclass_fields__.keys():
            if str(attr).endswith("_file"):
                if attr in model_files:
                    full_path = self._prepend_directory(attr, "model_directory")
                elif attr in data_files:
                    full_path = self._prepend_directory(attr, "data_directory")
                elif attr in log_files:
                    full_path = self._prepend_directory(attr, "log_directory")
                elif attr in attack_files:
                    full_path = self._prepend_directory(attr, "attack_directory")
                else:
                    full_path = getattr(self, attr)
                files[attr] = full_path
        return files
        
        
        