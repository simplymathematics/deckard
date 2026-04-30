import logging
import argparse
import inspect
import pandas as pd
import pickle
import json
import importlib
import sys
import traceback
import hashlib

from pathlib import Path
from typing import Union, Any
from dataclasses import dataclass, field
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _canonicalize_for_hash(value):
    """Convert arbitrary values into a stable, JSON-serializable structure."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        return {
            str(k): _canonicalize_for_hash(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_hash(v) for v in value]

    if isinstance(value, (set, frozenset)):
        items = [_canonicalize_for_hash(v) for v in value]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": bytes(value).hex()}

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _canonicalize_for_hash(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        public_attrs = {
            k: v
            for k, v in value.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
        return _canonicalize_for_hash(public_attrs)

    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"

    return str(value)


def normalize_for_hash(value, root=None):
    """Normalize values for stable hashing.

    Mirrors resolver behavior:
    - Optional key-path lookup when `value` is a string and `root` is provided.
    - OmegaConf nodes resolved to plain Python containers.
    """
    target = value

    if isinstance(value, str) and root is not None:
        selected = OmegaConf.select(root, value, default=None)
        if selected is not None:
            target = selected

    return _canonicalize_for_hash(target)


def hash_conf_values(*values, _root_=None) -> str:
    """Return stable MD5 hash for one or more config-like values.

    Supports the same patterns as the `${hash:...}` resolver:
    - no values: hash `_root_`
    - one value: hash normalized value
    - many values: hash normalized list of values in order
    """
    if not values:
        target = _root_
    elif len(values) == 1:
        target = normalize_for_hash(values[0], root=_root_)
    else:
        target = [normalize_for_hash(v, root=_root_) for v in values]

    s = json.dumps(target, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


data_supported_filetypes = [
    ".csv",
    ".parquet",
    ".pkl",
    ".html",
    ".json",
    ".xlsx",
    ".openml",
]


@dataclass
class ConfigBase:
    # _target_: str = "deckard.utils.ConfigBase"
    score_dict: dict = field(default_factory=dict)
    HASH_EXCLUDE_FIELDS = {
        "args",
        "score_dict",
        "predictions",
        "probabilities",
        "labels",
        "attack_predictions",
        "attack_probabilities",
        "adv_predictions",
        "adv_probabilities",
        "X",
        "y",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
    }
    HASH_EXCLUDE_SUFFIXES = (
        "_time",
        "_predictions",
        "_probabilities",
    )

    def __init__(self, *args, **kwds):
        # Initialize dataclass super
        super().__init__()

        # Initialize args attribute
        self.args = args if args else ()
        #  Set attributes from args and kwds
        for i, arg in enumerate(args):
            setattr(self, list(self.__dataclass_fields__.keys())[i], arg)
        for k, v in kwds.items():
            setattr(self, k, v)
        # Call post init
        self.__post_init__()
        # Freeze hash at configuration time so runtime attributes added during
        # execution cannot alter experiment identity.
        self._hash_payload = self.to_dict(for_hash=True)
        self._hash_value = hash_conf_values(self._hash_payload)

    def __post_init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("This is an abstract base class.")

    def __hash__(self):
        """Return the initialization-time configuration hash as int."""
        if "_hash_value" not in self.__dict__:
            self._hash_payload = self.to_dict(for_hash=True)
            self._hash_value = hash_conf_values(self._hash_payload)
        return int(self._hash_value, 16)

    def _is_hash_field(self, name: str) -> bool:
        if name == "_target_":
            return True
        if name.startswith("_"):
            return False
        if name in self.HASH_EXCLUDE_FIELDS:
            return False
        if any(name.endswith(suffix) for suffix in self.HASH_EXCLUDE_SUFFIXES):
            return False
        return True

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

    def read_or_initialize_scores(self, score_file):
        """Return merged scores from disk and memory, or initialize output location.

        This is the canonical entrypoint for score-file reads in ConfigBase.
        """
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
                case ".xlsx":
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

    def to_yaml(self) -> str:
        """
        Converts the current instance to a YAML string.

        Returns
        -------
        str
            A YAML representation of the instance.
        """
        config = self.to_dict()
        config = OmegaConf.create(config)
        return str(OmegaConf.to_yaml(config))

    def to_dict(self, for_hash: bool = False) -> dict:
        """
        Converts the current instance to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the instance.
        """
        # Build a dict from inherited dataclass fields + runtime attributes
        dict_ = {}

        # Include dataclass fields from full MRO (base -> child)
        for base in reversed(self.__class__.mro()):
            fields = getattr(base, "__dataclass_fields__", {})
            for name in fields:
                if name.startswith("_") and not (for_hash and name == "_target_"):
                    continue
                if for_hash and not self._is_hash_field(name):
                    continue
                if hasattr(self, name):
                    value = getattr(self, name)
                    if isinstance(value, ConfigBase):
                        dict_[name] = value.to_dict(for_hash=for_hash)
                    elif OmegaConf.is_config(value):
                        dict_[name] = OmegaConf.to_container(value, resolve=True)
                    else:
                        dict_[name] = value

        # Include any additional runtime attrs not declared as dataclass fields
        for name, value in self.__dict__.items():
            if (name.startswith("_") and not (for_hash and name == "_target_")) or name in dict_:
                continue
            if for_hash and not self._is_hash_field(name):
                continue
            if isinstance(value, ConfigBase):
                dict_[name] = value.to_dict(for_hash=for_hash)
            elif OmegaConf.is_config(value):
                dict_[name] = OmegaConf.to_container(value, resolve=True)
            else:
                dict_[name] = value

        return dict_

    def execute_without_mercy(self):
        # Get log_file from logger
        log_file = next(
            (
                handler.baseFilename
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler)
            ),
            "deckard.log",
        )
        try:
            scores = self()
        except Exception as e:
            with open(log_file, "+a") as log_f:
                tb = traceback.format_exc()
                log_f.write(f"\nException: {e}\n")
                log_f.write(tb)
                log_f.write("\n")
            logger.error(e)
            if hasattr(self, "score_dict"):
                scores = self.score_dict
            else:
                scores = {}
        return scores


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
        case ".html":
            data = pd.read_html(filepath, **kwargs)[0]
        case _:
            raise ValueError(
                f"Unsupported file type {Path(filepath).suffix}. Supported types: {supported_filetypes}",
            )
    logger.info(f"Data loaded from {Path(filepath)}")
    return data


def import_class_from_file(
    file_path: str,
    class_name: str,
    *args,
    instantiate_class: bool = True,
    **kwargs,
):
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
    if not instantiate_class:
        return cls
    return cls(*args, **kwargs)


def resolve_class(cls: str):
    """Resolve a class path into a class object without instantiating it.

    Supports dotted module paths (Hydra-style) and ``file.py:ClassName`` paths.
    """
    if not isinstance(cls, str):
        raise TypeError(f"Class path must be a string. Got {type(cls)}")

    if ":" in cls:
        file_path, class_name = cls.split(":", 1)
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        return import_class_from_file(
            file_path,
            class_name,
            instantiate_class=False,
        )

    try:
        return get_class(cls)
    except Exception:
        module_name, class_name = cls.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


def load_class(cls, *args, **kwargs):
    if isinstance(cls, type):
        return cls(*args, **kwargs)

    if not isinstance(cls, str):
        raise TypeError(f"Class path must be a string. Got {type(cls)}")

    if ":" in cls:
        class_obj = resolve_class(cls)
        return class_obj(*args, **kwargs)

    instantiate_kwargs = dict(kwargs)
    if args:
        instantiate_kwargs["_args_"] = list(args)
    return instantiate({"_target_": cls, **instantiate_kwargs})


def create_parser_from_function(
    func,
    parser=None,
    exclude=None,
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
    if exclude is None:
        exclude = []
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
