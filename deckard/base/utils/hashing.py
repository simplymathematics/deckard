from hashlib import md5
from collections import OrderedDict
from typing import NamedTuple, Union
from dataclasses import asdict, is_dataclass
from omegaconf import DictConfig, OmegaConf, SCMode, ListConfig
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def to_dict(obj: Union[dict, OrderedDict, NamedTuple]) -> dict:
    new = {}
    if hasattr(obj, "_asdict"):
        obj = obj._asdict()
    if isinstance(obj, dict):
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, OrderedDict):
        sorted_keys = obj
    elif isinstance(obj, (DictConfig, OmegaConf)):
        obj = dict(
            deepcopy(
                OmegaConf.to_container(
                    obj,
                    resolve=True,
                    structured_config_mode=SCMode.DICT,
                ),
            ),
        )
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, (ListConfig)):
        obj = deepcopy(OmegaConf.to_container(obj, resolve=True))
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif is_dataclass(obj):
        obj = deepcopy(asdict(obj))
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, str):
        obj = obj
        sorted_keys = []
    elif isinstance(obj, type(None)):
        obj = None
        sorted_keys = []
    elif is_dataclass(obj):
        obj = deepcopy(asdict(obj))
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    else:
        raise ValueError(
            f"obj must be a Dict, namedtuple or OrderedDict. It is {type(obj)}",
        )
    for key in sorted_keys:
        try:
            if obj[key] is None:
                continue
            elif isinstance(obj[key], (str, float, int, bool, tuple, list)):
                new[key] = obj[key]
            elif is_dataclass(obj[key]):
                new[key] = asdict(obj[key])
            elif isinstance(obj[key], (DictConfig)):
                new[key] = to_dict(obj[key])
            elif isinstance(obj[key], (ListConfig)):
                new[key] = OmegaConf.to_container(obj[key], resolve=True)
            elif isinstance(obj[key], (dict)):
                new[key] = to_dict(obj[key])
            elif isinstance(obj[key], (list, tuple)):
                new[key] = [to_dict(x) for x in obj[key]]
            else:
                new[key] = obj[key]
        except Exception as e:
            logger.error(f"Error while converting {key} to dict")
            logger.error(f"obj[key] = {obj[key]}")
            logger.error(f"type(obj[key]) = {type(obj[key])}")
            logger.error(f"obj = {obj}")
            raise e
    return new


def my_hash(obj: Union[dict, OrderedDict, NamedTuple]) -> str:
    return md5(str(to_dict(obj)).encode("utf-8")).hexdigest()
