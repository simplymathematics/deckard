from hashlib import md5
from numpy import ndarray
from typing import Callable


def my_hash(obj:dict):
    new = {}
    sorted_keys = list(obj.keys())
    sorted_keys.sort()
    for key in sorted_keys:
        if isinstance(key, dict):
            new[key] = my_hash(obj[key])
        else:
            new[key] = obj[key]
    return md5(str(new).encode()).hexdigest()


class BaseHashable(object):
    def __hash__(self) -> str:
        """
        Hashes the params as specified in the __init__ method.
        """
        return my_hash(self._asdict())
