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
    if isinstance(obj, OrderedDict):
        sorted_keys = obj
    elif isinstance(obj, dict):
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, (DictConfig)):
        obj = dict(
            deepcopy(
                OmegaConf.to_container(
                    obj, resolve=True, structured_config_mode=SCMode.DICT,
                ),
            ),
        )
        sorted_keys = list(obj.keys())
        sorted_keys.sort()
    elif isinstance(obj, (ListConfig)):
        obj = deepcopy(OmegaConf.to_container(obj, resolve=True))
        sorted_keys = range(len(obj))
        sorted_values = obj
        obj = OrderedDict(zip(sorted_keys, sorted_values))
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
        sorted_keys.sort()
    elif isinstance(obj, (list, tuple)):
        sorted_keys = range(len(obj))
        sorted_values = obj
        obj = OrderedDict(zip(sorted_keys, sorted_values))
    else:  # pragma: no cover
        raise ValueError(
            f"obj must be a Dict, namedtuple or OrderedDict. It is {type(obj)}",
        )
    for key in sorted_keys:
        try:
            if obj[key] is None:
                continue
            elif isinstance(obj[key], (str, float, int, bool, tuple, list)):
                new[key] = obj[key]
            else:
                new[key] = to_dict(obj[key])
        except Exception as e:  # pragma: no cover
            logger.error(f"Error while converting {key} to dict")
            logger.error(f"obj[key] = {obj[key]}")
            logger.error(f"type(obj[key]) = {type(obj[key])}")
            logger.error(f"obj = {obj}")
            raise e
    return new


def my_hash(obj: Union[dict, OrderedDict, NamedTuple]) -> str:
    return md5(str(to_dict(obj)).encode("utf-8")).hexdigest()
