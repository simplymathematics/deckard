import logging
import argparse
import inspect
import pandas as pd
import pickle
import json
import importlib
import sys
import os

from pathlib import Path
from hashlib import md5
from typing import Union, Any
from dataclasses import dataclass, field
from hydra.utils import instantiate
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


data_supported_filetypes = [
    ".csv",
    ".parquet",
    ".pkl",
    ".html",
    ".json",
    ".xlsx",
    ".pkl",
    ".openml",
]


@dataclass
class ConfigBase:
    # _target_: str = "deckard.utils.ConfigBase"
    score_dict :dict = field(default_factory=dict)

    def __init__(self, *args, **kwds):
        # Initialize dataclass super
        super().__init__()
        # Initialize args attribute
        self.args = args if args else ()
        # Call post init
        #  Set attributes from args and kwds
        for i, arg in enumerate(args):
            setattr(self, list(self.__dataclass_fields__.keys())[i], arg)
        for k, v in kwds.items():
            setattr(self, k, v)
        # Call post init
        self.__post_init__()

    def __post_init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("This is an abstract base class.")

    def __hash__(self):
        """
        Computes a hash value for the instance.

        Concatenates all non-private attribute names and values, then hashes the resulting string using MD5.
        The hash excludes attributes whose names start with an underscore.

        Returns
        -------
        int
            The integer representation of the MD5 hash of the concatenated attribute string.
        """
        # Hash all fields that do not start with an underscore
        hash_input = "".join(
            f"{k}:{v}" for k, v in self.__dict__.items() if not k.endswith("_")
        )
        return int(md5(hash_input.encode()).hexdigest(), 16)

    def save_scores(
        self,
        scores: Union[dict, pd.Series],
        filepath: Union[str, None] = None,
    ):
        """
        Saves the scores dictionary to a CSV file if a filepath is provided.

        Parameters
        ----------
        scores : dict
            Dictionary containing score metrics to be saved.
        filepath : Union[str, None], optional
            Path to save the scores as a CSV file. If None, scores are not saved.

        Raises
        ----------
        ValueError
            If the file extension is not supported. Supported types are .csv, .json, and .xlsx.
        """
        assert filepath is not None, "Filepath must be provided to save scores."
        score_path = Path(filepath)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        # Assume this is a dictionary of of strings: floats
        supported_filtypes = [".csv", ".json", ".xlsx"]
        if not isinstance(scores, pd.Series):
            scores = pd.Series(scores)
        if score_path.suffix in supported_filtypes:
            match score_path.suffix:
                case ".csv":
                    scores.to_csv(score_path, index=False)
                case ".json":
                    with open(score_path, "w") as f:
                        json.dump(scores.to_dict(), f, indent=4)
                case ".xlsx":
                    scores.to_excel(score_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filtypes}",
            )
        assert Path(score_path).exists(), f"Failed to save scores to {score_path}"
        logger.info(f"Scores saved to {score_path}")

    def save_data(
        self,
        data: pd.DataFrame,
        filepath: Union[str, None] = None,
        **kwargs,
    ) -> None:
        supported_filetypes = [
            ".csv",
            ".parquet",
            ".pkl",
            ".html",
            ".json",
            ".xlsx",
            ".pkl",
        ]
        assert filepath is not None, "Filepath must be provided to save data."
        data_path = Path(filepath)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        filetype = data_path.suffix
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        match filetype:
            case ".pkl":
                data.to_pickle(data_path, **kwargs)
            case ".csv":
                data.to_csv(data_path, index=False, **kwargs)
            case ".parquet":
                data.to_parquet(data_path, index=False, **kwargs)
            case ".pkl":
                data.to_pickle(data_path, **kwargs)
            case ".html":
                data.to_html(data_path, index=False, **kwargs)
            case ".json":
                data.to_json(data_path, orient="records", lines=True, **kwargs)
            case ".xlsx":
                data.to_excel(data_path, index=False, **kwargs)
            case _:
                raise ValueError(
                    f"Unsupported file type {data_path.suffix}. Supported types: {supported_filetypes}",
                )
        assert Path(data_path).exists(), f"Failed to save data to {data_path}"
        logger.info(f"Data saved to {data_path}")

    def read_scores_from_disk(self, score_file):
        if score_file is not None and Path(score_file).exists():
            # Load existing scores
            logger.info(f"Loading existing scores from {score_file}")
            disk_scores = self.load_scores(score_file)
            scores = {**self.score_dict, **disk_scores}
        elif score_file is not None:
            # Ensure directory exists
            logger.debug(f"Creating directory for scores at {score_file}")
            Path(score_file).parent.mkdir(parents=True, exist_ok=True)
            scores = self.score_dict
        else:
            logger.debug("No score_file provided, scores will not be saved")
            if hasattr(self, "score_dict"):
                scores = self.score_dict
            else:
                scores = {}
        return scores
    
    def get_call_params(self) -> dict:
        """
        Retrieves the parameters required to call the __call__ method of the instance.

        Returns
        -------
        dict
            A dictionary containing parameter names and their corresponding values.
        """
        sig = inspect.signature(self.__call__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
            else:
                raise AttributeError(
                    f"Instance of {self.__class__.__name__} does not have attribute {name} required for __call__",
                )
        return params

    def load_scores(self, filepath: str) -> dict:
        """
        Loads scores from a CSV, JSON, or Excel file into a dictionary.

        Parameters
        ----------
        filepath : str
            Path to the scores file.

        Returns
        -------
        dict
            Dictionary containing the loaded scores.

        Raises
        ------
        ValueError
            If the file extension is not supported. Supported types are .csv, .json, and .xlsx.
        """
        score_path = Path(filepath)
        assert score_path.exists(), f"File {filepath} does not exist."
        supported_filetypes = [".csv", ".json", ".xlsx"]
        if score_path.suffix in supported_filetypes:
            match score_path.suffix:
                case ".csv":
                    scores = pd.read_csv(score_path)
                case ".json":
                    with open(score_path, "r") as f:
                        scores = json.load(f)
                    
                    if "files" in scores:
                        files = scores.pop("files")
                    if "params" in scores:
                        params = scores.pop("params")
                    if "files" in locals():
                        scores["files"] = files
                    if "params" in locals():
                        scores["params"] = params
                case "xlsx":
                    scores = pd.read_excel(score_path)
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
            )
        logger.info(f"Scores loaded from {score_path}")
        return scores

    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        return load_data(filepath, **kwargs)

    def save_object(self, obj: Any, filepath: str) -> None:
        """
        Saves a Serializable object to a file using pickle.

        Parameters
        ----------
        obj : Any
            The object to save.
        filepath : str
            The path to the file where the object will be saved.
        Raises
        ------
        ValueError
            If the file extension is not supported. Supported types are .pkl and .pickle.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        suffix = Path(filepath).suffix
        supported_suffixes = [".pkl", ".pickle"]
        if suffix not in supported_suffixes:
            raise ValueError(
                f"Unsupported file type {suffix}. Supported types: {supported_suffixes}",
            )
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {filepath}")

    def load_object(self, filepath: str) -> Any:
        """
        Loads a Serializable object from a file using pickle.

        Parameters
        ----------
        filepath : str
            The path to the file from which the object will be loaded.

        Returns
        -------
        Any
            The loaded object.
        """
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from {filepath}")
        return obj

    def save(self, filepath: str) -> None:
        """
        Saves the current instance to a file using pickle.

        Parameters
        ----------
        filepath : str
            The path to the file where the instance will be saved.
        """
        if Path(filepath).exists():
            raise ValueError(f"File {filepath} already exists. Will not overwrite.")
        self.save_object(self, filepath)
        logger.info(f"Instance of {self.__class__.__name__} saved to {filepath}")

    def load(self, filepath: str) -> "ConfigBase":
        """
        Loads an instance of the class from a file using pickle.

        Parameters
        ----------
        filepath : str
            The path to the file from which the instance will be loaded.

        Returns
        -------
        ConfigBase
            The loaded instance.
        """
        assert Path(filepath).exists(), f"File {filepath} does not exist."
        obj = self.load_object(filepath)
        if not isinstance(obj, self.__class__):
            raise TypeError(f"Loaded object is not of type {self.__class__.__name__}")
        logger.info(f"Instance of {self.__class__.__name__} loaded from {filepath}")
        # Update the current instance's __dict__ with the loaded object's __dict__
        self.__dict__.update(obj.__dict__)
        return self

    @staticmethod
    def from_yaml(filepath: str) -> "ConfigBase":
        """
        Creates an instance of the class from a YAML configuration file.

        Parameters
        ----------
        filepath : str
            The path to the YAML configuration file.

        Returns
        -------
        ConfigBase
            An instance of the class initialized with the configuration from the YAML file.
        """
        config = OmegaConf.to_container(OmegaConf.load(filepath), resolve=True)
        if not isinstance(config, dict):
            raise TypeError(f"Loaded config is not a dictionary from {filepath}")
        instance = instantiate(config)
        logger.info(
            f"Instance of {instance.__class__.__name__} created from {filepath}",
        )
        return instance

    @staticmethod
    def from_dict(data: dict) -> "ConfigBase":
        """
        Creates an instance of the class from a dictionary.

        Parameters
        ----------
        data : dict
            The dictionary containing the configuration.

        Returns
        -------
        ConfigBase
            An instance of the class initialized with the configuration from the dictionary.
        """
        instance = instantiate(data)
        return instance

    @classmethod
    def from_yaml(cls, filepath: str = None, yaml_string: str = None) -> "ConfigBase":
        assert filepath is not None or yaml_string is not None, "Either filepath or yaml_string must be provided."
        if filepath is not None:
            return cls.from_yaml_file(filepath)
        else:
            return cls.from_yaml_string(yaml_string)
    
    @classmethod
    def to_yaml(cls) -> str:
        """
        Converts the current instance to a YAML string.

        Returns
        -------
        str
            A YAML representation of the instance.
        """
        config = cls.to_dict()
        config = OmegaConf.create(config)
        return str(OmegaConf.to_yaml(config))

    @classmethod 
    def to_dict(cls) -> dict:
        """
        Converts the current instance to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the instance.
        """
        # Inspect to find init parameters
        signature = inspect.signature(cls.__init__)
        dict_ = {}
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if hasattr(cls, name):
                value = getattr(cls, name)
                dict_[name] = value
        return dict_
    
    def save_data(
            self,
            data: pd.DataFrame,
            filepath: Union[str, None] = None,
            **kwargs,
        ) -> None:
        save_data(data, filepath, **kwargs)
        
        
def save_data(
    data: pd.DataFrame,
    filepath: Union[str, None] = None,
    **kwargs,
) -> None:
    supported_filetypes = [
        ".csv",
        ".parquet",
        ".pkl",
        ".html",
        ".json",
        ".xlsx",
        ".pkl",
    ]
    assert filepath is not None, "Filepath must be provided to save data."
    data_path = Path(filepath)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    filetype = data_path.suffix
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    match filetype:
        case ".pkl":
            data.to_pickle(data_path, **kwargs)
        case ".csv":
            data.to_csv(data_path, index=False, **kwargs)
        case ".parquet":
            data.to_parquet(data_path, index=False, **kwargs)
        case ".pkl":
            data.to_pickle(data_path, **kwargs)
        case ".html":
            data.to_html(data_path, index=False, **kwargs)
        case ".json":
            data.to_json(data_path, orient="records", lines=True, **kwargs)
        case ".xlsx":
            data.to_excel(data_path, index=False, **kwargs)
        case _:
            raise ValueError(
                f"Unsupported file type {data_path.suffix}. Supported types: {supported_filetypes}",
            )
    assert Path(data_path).exists(), f"Failed to save data to {data_path}"
    logger.info(f"Data saved to {data_path}")

def load_data(filepath: str, **kwargs) -> pd.DataFrame:
        """
        Loads data from a CSV, JSON, Excel, Parquet, Pickle, NPZ, or HTML file into a pandas DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the data file.
        **kwargs
            Additional keyword arguments to pass to the pandas read function.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the loaded data.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If the file extension is not supported. Supported types are .csv, .json, .
        """

        if filepath is None:
            raise FileNotFoundError("Filepath is None.")
        supported_filetypes = [
            ".csv",
            ".json",
            ".xlsx",
            ".parquet",
            ".pkl",
            ".npz",
            ".html",
        ]

        match Path(filepath).suffix:
            case ".pkl":
                data = pd.read_pickle(filepath, **kwargs)
            case ".csv":
                data = pd.read_csv(filepath, **kwargs)
            case ".json":
                data = pd.read_json(filepath, orient="records", **kwargs)
            case ".xlsx":
                data = pd.read_excel(filepath, **kwargs)
            case ".parquet":
                data = pd.read_parquet(filepath, **kwargs)
            case "html":
                data = pd.read_html(filepath, **kwargs)[0]
            case _:
                raise ValueError(
                    f"Unsupported file type {Path(filepath).suffix}. Supported types: {supported_filetypes}",
                )
        logger.info(f"Data loaded from {Path(filepath)}")
        return data


def import_class_from_file(file_path: str, class_name: str, *args, **kwargs):
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    spec.loader.exec_module(module)

    cls = getattr(module, class_name)
    return cls(*args, **kwargs)


def load_class(cls, *args, **kwargs):
    if ":" in cls:
        file_path, class_name = cls.split(":", 1)
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        cls = import_class_from_file(file_path, class_name, *args, **kwargs)
    else: 
        if not Path(cls).exists():
            try:
               _ = importlib.import_module(cls.split(".")[0]) 
            except ImportError as e:
                raise ImportError(e)         
        cls = instantiate({"_target_": path, **kwargs})
    return cls

def create_parser_from_function(
    func,
    parser=None,
    exclude=[],
    **kwargs,
) -> argparse.ArgumentParser:
    """
    Creates an argparse.ArgumentParser from a function's signature.

    Parameters
    ----------
    func: callable
        The function to create the parser from.
    parser : argparse.ArgumentParser, optional
        An existing parser to add arguments to. If None, a new parser is created.
    exclude: list, optional
        List of parameter names to exclude from the parser.
    **kwargs
        Additional keyword arguments to pass to the ArgumentParser constructor if a new parser is created.

    Raises
    ------
    ValueError
        If func is not callable or if parser is not an instance of argparse.ArgumentParser.


    Returns
    -------
    argparse.ArgumentParser
        The updated parser with arguments corresponding to the function's signature.
    """
    # Validate the parser
    conflict_handler = kwargs.pop("conflict_handler", "resolve")
    add_help = kwargs.pop("add_help", False)
    if parser is None:
        parser = argparse.ArgumentParser(
            **kwargs,
            conflict_handler=conflict_handler,
            add_help=add_help,
        )
    else:
        if len(kwargs) > 0:
            raise ValueError("Cannot pass kwargs when parser is provided.")
        if not isinstance(parser, argparse.ArgumentParser):
            raise ValueError(
                f"parser must be an instance of argparse.ArgumentParser or None. Got {type(parser)}",
            )
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if name == "self" or name in exclude:
            continue
        if param.annotation is not inspect._empty:
            arg_type = param.annotation
        else:
            arg_type = str  # Default to string if no annotation
        if param.default is inspect._empty:
            parser.add_argument(f"--{name}", type=arg_type, required=True)
        else:
            parser.add_argument(f"--{name}", type=arg_type, default=param.default)
    return parser
