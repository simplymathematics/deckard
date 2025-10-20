import logging
import argparse
import inspect
from pathlib import Path
from hashlib import md5
from typing import Union, Any
from dataclasses import dataclass
import pandas as pd
import pickle
import os
from hydra import initialize, compose
from hydra.utils import instantiate


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




def initialize_config(config_file, params, target, **kwargs) -> object:
    """
    Initializes and composes a Hydra configuration.

    Depending on the provided arguments, this function loads a configuration file,
    applies parameter overrides, and ensures the object_name is set in the configuration.

    Args:
        config_file (str or list or None): Path to the configuration file, or a list of override strings.
        params (list or None): List of parameter overrides in the format ["key=value", ...].
        object_name (str): The object_name string to be included in the configuration if not already present.
        **kwargs: Additional key-value pairs to be added to params.

    Returns:
        DictConfig: The composed Hydra configuration object.

    Raises:
        AssertionError: If the overrides or params are not provided as lists.
    """
    if len(kwargs) > 0:
        params = {**params, **kwargs} if params is not None else kwargs
    if config_file and not params:
        logger.info(f"Loading config from {config_file}")
        folder = str(Path(config_file).parent)
        assert Path(folder).exists(), f"Config folder {folder} does not exist. Current working directory: {os.getcwd()}"
        filename = str(Path(config_file).name)
        with initialize(config_path=folder):
            config = compose(config_name=filename)
    elif config_file:
        logger.info(f"Overriding config from {config_file} with params:")
        override_config = config_file
        keys = [k.split("=")[0] for k in override_config]
        for param in params:
            logger.info(f" - {param}")
            override_config.append(param)
        if "_target_" not in keys:
            override_config = [f"++_target_={target}"] + override_config
        assert isinstance(
            override_config,
            list,
        ), "config must be a YAML list of dictionaries"
        if config_file is None:
            with initialize(config_path=None):
                config = compose(config_name=None, overrides=override_config)
        else:
            folder = str(Path(config_file).parent)
            filename = str(Path(config_file).name)
            with initialize(config_path=folder):
                config = compose(config_name=filename, overrides=override_config)
    else:
        params = params if params is not None else []
        keys = [k.split("=")[0] for k in params]
        if "_target_" not in keys:
            params = [f"++_target_={target}"] + params
        with initialize(config_path=None):
            config = compose(config_name=None, overrides=params)
    obj = instantiate(config)
    # Change back to original working directory
    return obj


@dataclass
class ConfigBase:       
    _target_: str = "deckard.utils.ConfigBase"

    def __init__(self, *args, **kwds):
        # Initialize dataclass super
        super().__init__()
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

    def __call__(self, *args, **kwds):
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
            f"{k}:{v}" for k, v in self.__dict__.items() if not k.startswith("_")
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
                    scores.to_json(score_path, indent=4)
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
                    scores = pd.read_json(
                        score_path,
                        typ="series",
                    )
                case "xlsx":
                    scores = pd.read_excel(score_path)
        else:
            raise ValueError(
                f"Unsupported file type {score_path.suffix}. Supported types: {supported_filetypes}",
            )
        logger.info(f"Scores loaded from {score_path}")
        return scores

    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
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
        data_path = Path(filepath)
        if data_path is None or not data_path.exists():
            FileNotFoundError(f"File {filepath} does not exist.")
        supported_filetypes = [
            ".csv",
            ".json",
            ".xlsx",
            ".parquet",
            ".pkl",
            ".npz",
            ".html",
        ]

        match data_path.suffix:
            case ".pkl":
                data = pd.read_pickle(data_path, **kwargs)
            case ".csv":
                data = pd.read_csv(data_path, **kwargs)
            case ".json":
                data = pd.read_json(data_path, orient="records", **kwargs)
            case ".xlsx":
                data = pd.read_excel(data_path, **kwargs)
            case ".parquet":
                data = pd.read_parquet(data_path, **kwargs)
            case "html":
                data = pd.read_html(data_path, **kwargs)[0]
            case _:
                raise ValueError(
                    f"Unsupported file type {data_path.suffix}. Supported types: {supported_filetypes}",
                )
        logger.info(f"Data loaded from {data_path}")
        return data

    def save_object(self, obj: Any, filepath: str) -> None:
        """
        Saves a Serializable object to a file using pickle.

        Parameters
        ----------
        obj : Any
            The object to save.
        filepath : str
            The path to the file where the object will be saved.
        """
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
        obj = self.load_object(filepath)
        if not isinstance(obj, self.__class__):
            raise TypeError(f"Loaded object is not of type {self.__class__.__name__}")
        logger.info(f"Instance of {self.__class__.__name__} loaded from {filepath}")
        # Update the current instance's __dict__ with the loaded object's __dict__
        self.__dict__.update(obj.__dict__)
        return self


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
    # Validate that the func is callable
    assert callable(func), "func must be a callable function or method."
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
                "parser must be an instance of argparse.ArgumentParser or None.",
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
