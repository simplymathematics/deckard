import json
import yaml
from hashlib import md5
from typing import Union, Any, NamedTuple
import collections
import logging
from pathlib import Path
import tempfile
from time import time
from sklearn.model_selection import ParameterGrid
from pandas import DataFrame, Series, read_csv


def to_dict(obj: Union[dict, collections.OrderedDict, NamedTuple]) -> dict:
    new = {}
    if hasattr(obj, "_asdict"):
        obj = obj._asdict()
    if isinstance(obj, dict):
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, collections.OrderedDict):
        sorted_keys = obj
    else:
        raise ValueError(
            f"obj must be a Dict, collections.namedtuple or collections.OrderedDict. It is {type(obj)}",
        )
    for key in sorted_keys:
        if isinstance(key, (dict, collections.OrderedDict)):
            new[key] = my_hash(obj[key])
        else:
            new[key] = obj[key]
    return new


my_hash = lambda obj: md5(str(to_dict(obj)).encode("utf-8")).hexdigest()

logger = logging.getLogger(__name__)


class BaseHashable:
    def __new__(cls, loader, node, *args, **kwds):
        return super().__new__(cls, **loader.construct_mapping(node))

    def __hash__(self):
        return int(my_hash(self), 32)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()})"

    def to_dict(self) -> dict:
        """Converts the object to a dictionary
        :return: dict representation of the object
        """
        logger.debug(f"Converting {self.__class__.__name__} to dict")
        return to_dict(self)

    def to_json(self) -> str:
        """Converts the object to a json string
        :return: json representation of the object"""
        logger.debug(f"Converting {self.__class__.__name__} to json")
        return json.dumps(self.to_dict())

    def to_yaml(self) -> str:
        """Converts the object to a yaml string
        :return: yaml representation of the object"""
        logger.debug(f"Converting {self.__class__.__name__} to yaml")
        return yaml.dump(self.to_dict())

    def save(self):
        """Saves the object to the files specified in the files attribute

        Raises:
            NotImplementedError: Needs to be implemented in the child class
        """
        raise NotImplementedError(
            f"No save method defined for {self.__class__.__name__}",
        )

    # def load(self):
    #     """Loads the object from the files specified in the files attribute

    #     Raises:
    #         NotImplementedError: Needs to be implemented in the child class
    #     """
    #     raise NotImplementedError(f"No load method defined for {self.__class__.__name__}")

    def save_yaml(self, path: Union[str, Path], filetype: str = "yaml"):
        if filetype.startswith("."):
            filetype = filetype[1:]
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = Path(path) / Path(my_hash(self) + "." + filetype)
        with filename.open("w") as f:
            f.write(self.to_yaml())
        logger.info(f"Saved {self.__class__.__name__} to {filename}")
        return filename

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__class__(*args, **kwds).load()

    def set_param(self, key_list, value, delimiter="."):
        """Sets a parameter in the object
        :param key_list: the key to set
        :param value: the value to set
        """
        params = self.to_dict()
        # sub_key = key.split(delimiter)
        cmd = f"params"
        if isinstance(key_list, str):
            key_list = key_list.split(delimiter)
        if not isinstance(key_list, list):
            key_list = [key_list]
        for sub in key_list:
            cmd += f"['{sub}']"
        cmd += f" = value"
        exec(cmd, locals())
        filename = tempfile.mktemp()
        with open(filename, "w") as f:
            yaml.dump(params, f)
        new = from_yaml(self, filename)
        return new


def from_yaml(hashable: BaseHashable, filename: Union[str, Path], key=None) -> Any:
    """Converts a yaml file to an object
    :param filename: path to the yaml file
    :return: object representation of the yaml file"""
    logger.debug(f"Loading {hashable.__class__.__name__}")
    yaml.add_constructor(f"!{hashable.__class__.__name__}\n", hashable.__class__)
    with Path(filename).open("r") as f:
        result = str(f.read())
        if not str(result).startswith(f"!{hashable.__class__.__name__}\n"):
            try:
                result = f"!{hashable.__class__.__name__}\n" + str(eval(result))
            except SyntaxError:
                result = f"!{hashable.__class__.__name__}\n" + result
        result = yaml.load(result, Loader=yaml.FullLoader)
    if key is not None:
        result = result[key]
    assert isinstance(
        result,
        hashable.__class__,
    ), f"Loaded object is not of type {hashable.__class__.__name__}. It is {type(result)}"
    return result


def from_dict(hashable: BaseHashable, config: dict) -> Any:
    """Converts a dictionary to an object
    :param params: dictionary to convert
    :return: object representation of the dictionary"""
    name = hashable.__class__.__name__
    logger.debug(f"Loading {name}")
    yaml.add_constructor(f"!{name}\n", hashable)
    result = yaml.load(f"!{name}\n" + yaml.dump(config), Loader=yaml.FullLoader)
    return result


def generate_line_search(hashable, param_name, param_list):
    """Generates a list of experiments with a line search
    :param param_name: the name of the parameter to search
    :param start: the start value of the parameter
    :param stop: the stop value of the parameter
    :param num: the number of values to search
    :param log: whether to use a logarithmic scale
    :return: a list of experiments
    """
    params = hashable.to_dict()
    new_param_list = []
    for entry in param_list:
        new_param_list.append(hashable.set_param(param_name, entry))
    return new_param_list


def generate_grid_search(hashable, param_dict):
    """Generates a list of experiments with a grid search
    :param param_dict: a dictionary of parameters to search
    :return: a list of experiments
    """
    params = hashable.to_dict()
    new_param_list = []
    for entry in ParameterGrid(param_dict):
        for key in entry:
            exp = hashable.set_param(key, entry[key])
        new_param_list.append(exp)
    return new_param_list


def generate_queue(
    hashable,
    param_dict,
    path: Union[str, Path] = "queue",
    filename="queue.csv",
) -> Path:
    """Generates a queue of experiments in specified path using a grid search
    :hashable: the hashable object to generate the queue for
    :param param_dict: a dictionary of parameters to search
    :param path: the path to save the queue to
    :return: the path to the queue
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    exp_list = generate_grid_search(hashable, param_dict)
    files = []
    hashes = []
    df = DataFrame()
    for exp in exp_list:
        files.append(exp.save_yaml(path=path))
        hashes.append(my_hash(exp))
        ser = Series(
            {
                "hash": hashes[-1],
                "file": files[-1],
                "status": "queued",
                "time": time(),
                "params": exp.to_dict(),
            },
        )
        df = df.append(ser, ignore_index=True)
    if Path(path, filename).exists():
        old_df = read_csv(Path(path) / Path(filename))
        df = old_df.append(df, ignore_index=True)
    df = df.drop_duplicates(subset="hash")
    df.to_csv(Path(path) / Path(filename))
    return Path(path) / Path(filename)


def sort_queue(
    path: Union[str, Path] = "queue",
    filename="queue.csv",
    by=["status", "time"],
) -> Path:
    """Sorts the queue by status
    :param path: the path to save the queue to
    :return: the path to the queue
    """
    df = read_csv(Path(path) / Path(filename))
    df = df.sort_values(by=by)
    df.to_csv(Path(path) / Path(filename))
    return Path(path) / Path(filename)
