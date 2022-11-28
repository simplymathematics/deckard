import json
import yaml
from hashlib import md5
from typing import  Union,  Any, NamedTuple
import collections
import logging
from pathlib import Path
import tempfile

def to_dict(obj: Union[dict, collections.OrderedDict]) -> dict:
    new = {}
    if hasattr(obj, "_asdict"):
        obj = obj._asdict()
    if isinstance(obj, dict):
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, collections.OrderedDict):
        sorted_keys = obj
    else:
        raise ValueError(f"obj must be a Dict, collections.namedtuple or collections.OrderedDict. It is {type(obj)}")
    for key in sorted_keys:
        if isinstance(key, (dict, collections.OrderedDict)):
            new[key] = my_hash(obj[key])
        else:
            new[key] = obj[key]
    return new



my_hash = lambda obj: md5(str(to_dict(obj)).encode("utf-8")).hexdigest()

logger = logging.getLogger(__name__)

class BaseHashable(collections.namedtuple(
    typename="BaseHashable", 
    field_names = "files",
    defaults = ({},)
)):
    
    def __hash__(self):
        return int(my_hash(self), 32)
    
    def __new__(cls, loader, node):
        return super().__new__(cls, **loader.construct_mapping(node))

    
    
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
        raise NotImplementedError(f"No save method defined for {self.__class__.__name__}")
   

                     
    def load(self):
        """Loads the object from the files specified in the files attribute

        Raises:
            NotImplementedError: Needs to be implemented in the child class
        """
        raise NotImplementedError(f"No load method defined for {self.__class__.__name__}")
    
    def save_yaml(self):
        files = dict(self.files)
        path_key = [key for key in files.keys() if "path" in key][0]
        filetype_key = [key for key in files.keys() if "file" in key][0]
        path = files.pop(path_key, "params")
        Path(path).mkdir(parents=True, exist_ok=True)
        filetype = files.pop(filetype_key, ".yaml")
        filename = Path(path) / my_hash(self) + "." + filetype
        with filename.open("w") as f:
            f.write(self.to_yaml())
        logger.info(f"Saved {self.__class__.__name__} to {filename}")
        return filename
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__class__(*args, **kwds).load()
    
    def set_param(self, key_list, value, delimiter = "."):
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
    
    def set_params(self, **kwargs):
        """Sets multiple parameters in the object
        :param kwargs: the key-value pairs to set
        """
        logger.debug(f"Setting {kwargs}")
        params = self.to_dict()
        for key, value in kwargs.items():
            self = self.set_param(key, value)
        return self

def from_yaml(hashable:BaseHashable, filename: Union[str, Path]) -> Any:
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
    assert isinstance(result, hashable.__class__), f"Loaded object is not of type {hashable.__class__.__name__}. It is {type(result)}"
    return result